#!/usr/bin/env python
# coding: utf-8

# ## Solving Laplace Equation using Finite Differences
# Implementing Jacobi and Gauss-Seidel Method. Better ones definitely exist! https://surface.syr.edu/cgi/viewcontent.cgi?article=1160&context=eecs_techreports

# In[2]:


import numpy as np 
from numerical_utils import gauss_seidel, initialize_solution
from matplotlib import pyplot as plt  
import time 

# Use Gauss seidel w/ large grid size and n_iters to get a good ground truth accuracy
n = 100
n_iters = 5000
boundaries = [0, 0, 100, 0] # [top, bottom, left, right]
solution = initialize_solution(n, boundaries)

# Run Gauss Seidel
tic = time.time()
solution = gauss_seidel(solution, n_iters)
toc = time.time()

print(f"Gauss Seidel took {toc - tic} seconds to run {n_iters} iterations")
ground_truth = np.copy(solution)


# In[3]:


plt.imshow(solution)


# In[5]:


from numerical_utils import jacobi, relative_l2_loss
# Experiment with runtimes and accuracies

# Jacobi
iterations = [100, 200, 400, 800, 1600, 3200]
jacobi_runtimes = []
jacobi_accuracies = []

for n_iters in iterations:
    solution = initialize_solution(n, boundaries)
    tic = time.time()
    solution = jacobi(solution, n_iters)
    toc = time.time()
    jacobi_runtimes.append(toc - tic)
    jacobi_accuracies.append(relative_l2_loss(solution, ground_truth))
    print(f"Jacobi took {toc - tic} seconds to run {n_iters} iterations with relative L2 loss {jacobi_accuracies[-1]}")


# In[6]:


# Gauss Seidel

gauss_seidel_runtimes = []
gauss_seidel_accuracies = []

for n_iters in iterations:
    solution = initialize_solution(n, boundaries)
    tic = time.time()
    solution = gauss_seidel(solution, n_iters)
    toc = time.time()
    gauss_seidel_runtimes.append(toc - tic)
    gauss_seidel_accuracies.append(relative_l2_loss(solution, ground_truth))
    print(f"Gauss Seidel took {toc - tic} seconds to run {n_iters} iterations with relative L2 loss {gauss_seidel_accuracies[-1]}")


# In[12]:


fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(iterations, jacobi_runtimes, label="Jacobi")
axs[0].plot(iterations, gauss_seidel_runtimes, label="Gauss Seidel")
axs[0].set_title("Runtimes")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Runtime (s)")
axs[0].legend()

axs[1].plot(iterations, jacobi_accuracies, label="Jacobi")
axs[1].plot(iterations, gauss_seidel_accuracies, label="Gauss Seidel")
axs[1].set_title("Relative L2 Loss")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Relative L2 Loss")
axs[1].legend()

