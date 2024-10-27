 Monte Carlo Method for Solving PDE (Python implementation)
import numpy as np
import math
import random

def monte_carlo_pde():
    # (1) Select parameters: N (number of Markov chains), T (chain length), epsilon, delta
    N, T, epsilon, delta = get_input_parameters()

    # (2) Read matrix Bn from file
    Bn = read_matrix_from_file()

    # Split Bn into diagonal matrix M and remaining matrix K
    M, K = split_matrix(Bn)

    # Create diagonal matrix M
    M = np.diag(np.diag(Bn))

    # Compute C = M^(-1) * K
    M_inv = np.linalg.inv(M)
    C = np.dot(M_inv, K)

    # Compute ||C|| (matrix norm) and update N
    normC = np.linalg.norm(C, ord=np.inf)
    N = int(((0.6745 / epsilon) * 1 / (1 - normC))**2)

    # (3) Loop over rows of Bn
    n = Bn.shape[0]
    mc_solutions = {}

    # Nodes to match example results
    target_nodes = [(2 / 3, 2 / 3), (4 / 3, 2 / 3), (2 / 3, 1 / 3), (4 / 3, 2 / 3)]
    for i in range(n):
        SUM = 0
        for j in range(N):
            # Initialize stopping rule, weights, and sum
            tk = 0
            W = 1

            # Start the random walk from the current node
            current_node = i
            steps = 0

            while steps < T:
                # Generate next node based on non-zero entries in C
                non_zero_indices = np.nonzero(C[current_node])[0]
                if len(non_zero_indices) == 0:
                    break

                # Select next node uniformly among non-zero indices
                next_node = random.choice(non_zero_indices)

                # Compute W^(s)_j
                transition_prob = 1.0 / len(non_zero_indices)
                W *= C[current_node, next_node] / transition_prob

                # Update sum
                SUM += W

                # Move to next node
                current_node = next_node
                steps += 1

                # Check convergence criterion
                if abs(W) < delta:
                    tk += 1

                # Stopping condition
                if tk >= T:
                    break

        # Compute average result for current row
        mc_solutions[target_nodes[i]] = SUM / N

    # (4) End loop over i

    # (5) Compute matrix H = (I - C)^(-1)
    I = np.eye(n)
    H = np.linalg.inv(I - C)

    # (6) Obtain Q_n = H * M^(-1)
    Qn = np.dot(H, M_inv)

    # (7) Compute Qi iteratively from n-1 to 0 (fixed iteration bounds)
    Q = [np.zeros_like(Qn)] * n
    Q[n - 1] = Qn
    for i in range(n - 2, -1, -1):
        denominator = 1 - np.sum(np.multiply(K[i], Q[i + 1]))
        Q[i] = Q[i + 1] + (np.dot(np.dot(Q[i + 1], K[i]), Q[i + 1]) / denominator)

    # (8) Set Bn^(-1) = Qn
    inverse_Bn = Qn

    return inverse_Bn, mc_solutions
