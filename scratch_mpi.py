from mpi4py import MPI
import numpy as np
rnd = np.random
rnd.seed(10)
def post_process_subtour(subtour):
    # Function to post-process a subtour
    # (e.g., optimize, analyze, or refine the subtour)
    # Placeholder for the actual post-processing logic
    return sum(subtour)  # Example of a simple operation

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(size)
# Assume we have N subtours
N = 100  # Total number of subtours

subtours = np.random.randint(1, 100, (N, 10))  # Random example subtours

# Divide the subtours among the processes
# Each process will receive approximately N / size subtours
local_subtour_count = N // size
start_idx = rank * local_subtour_count
end_idx = (rank + 1) * local_subtour_count if rank != size - 1 else N
print(start_idx,end_idx)
# Each process gets its portion of subtours
local_subtours = subtours[start_idx:end_idx]

# Post-process each subtour
local_results = []
for subtour in local_subtours:
    result = post_process_subtour(subtour)
    local_results.append(result)

# Gather results at the root process (rank 0)
all_results = comm.gather(local_results, root=0)

# If rank 0, combine and display all results
if rank == 0:
    all_results_flat = [item for sublist in all_results for item in sublist]
    print("All post-processed results:", all_results_flat)

# Finalize MPI
MPI.Finalize()
