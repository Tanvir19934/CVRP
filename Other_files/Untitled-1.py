import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV



# Load dataset
file_path = '/Users/tanvirkaisar/Downloads/hospital_readmissions.csv'
data = pd.read_csv(file_path)

# One-hot encoding categorical variables
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
onehot_encoder = OneHotEncoder(sparse=False)
encoded_categorical_data = onehot_encoder.fit_transform(data[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical data with numerical data
numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
numerical_data = data[numerical_columns]
preprocessed_data = pd.concat([numerical_data, encoded_categorical_df], axis=1)

# Label encoding for the target variable
label_encoder = LabelEncoder()
data['readmitted'] = label_encoder.fit_transform(data['readmitted'])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, data['readmitted'], test_size=0.2, random_state=42)

# Define the models
random_forest = RandomForestClassifier(random_state=42)
adaboost = AdaBoostClassifier(random_state=42,learning_rate=0.1)
ann = MLPClassifier(random_state=42, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
  # Increased max_iter for convergence

# Train
random_forest.fit(X_train, y_train)
adaboost.fit(X_train, y_train)
ann.fit(X_train, y_train)

# Make prediction
rf_pred = random_forest.predict(X_test)
ab_pred = adaboost.predict(X_test)
ann_pred = ann.predict(X_test)

# evaluation metrics
models = [random_forest, adaboost, ann]
predictions = [rf_pred, ab_pred, ann_pred]
model_names = ['Random Forest', 'AdaBoost', 'ANN']
results = {}
for model, pred, name in zip(models, predictions, model_names):
    results[name] = {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'ROC AUC': roc_auc_score(y_test, pred),
        'Confusion Matrix': confusion_matrix(y_test, pred)
    }

# Print the results
for model_name, metrics in results.items():
    print(f"Results for {model_name}:")
    for metric_name, metric_value in metrics.items():
        if metric_name == 'Confusion Matrix':
            print(f"{metric_name}:\n{metric_value}\n")
        else:
            print(f"{metric_name}: {metric_value:.4f}")


#plot the predictions in a subplot

predictions = {
    'AdaBoost': ab_pred,
    'Random Forest': rf_pred,
    'ANN': ann_pred
}

# Initialize the plot with 3x2 subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Loop through each model and plot the results
for i, (model_name, y_pred) in enumerate(predictions.items()):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Calculate the ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plotting the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'], ax=axes[i, 0])
    axes[i, 0].set_title(f'{model_name} - Confusion Matrix')
    axes[i, 0].set_xlabel('Predicted Label')
    axes[i, 0].set_ylabel('True Label')
    
    # Plotting the ROC Curve
    axes[i, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[i, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[i, 1].set_xlim([0.0, 1.0])
    axes[i, 1].set_ylim([0.0, 1.05])
    axes[i, 1].set_xlabel('False Positive Rate')
    axes[i, 1].set_ylabel('True Positive Rate')
    axes[i, 1].set_title(f'{model_name} - ROC Curve')
    axes[i, 1].legend(loc="lower right")

# Adjust the layout
plt.tight_layout()
# Display the plots
plt.show()

# Print the accuracy for each model
for model_name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {model_name}: {accuracy:.4f}')



###########Grid search of random forest###########


# Define the model
rf = RandomForestClassifier(random_state=42)

# Set up the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 10],
    'max_features': ['sqrt', 'log2']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit grid_search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on the test set
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")
