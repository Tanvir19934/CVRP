import torch
import matplotlib.pyplot as plt

# Create a sample 2D tensor (28x28 image)
h = torch.randn(28, 28)
print(h)

# Convert the tensor to a NumPy array explicitly
h_numpy = (h.abs() > 0.99).numpy()


# Plot the 2D data
plt.imshow(h_numpy, cmap='gray', interpolation='nearest')
plt.title('Sample 2D Tensor Data (Explicit Conversion)')
plt.show()
