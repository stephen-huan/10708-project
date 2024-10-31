import numpy as np


def relative_l2_loss(solution, solution_ref):
    """
    Compute the relative L2 loss between the solution and the reference solution.
    Args:
        solution: computed solution
        solution_ref: reference solution
    Returns:
        relative_l2_loss: relative L2 loss
    """
    numerator = np.linalg.norm(solution - solution_ref)
    denominator = np.linalg.norm(solution_ref)
    relative_l2_loss = numerator / denominator

    return relative_l2_loss


def initialize_solution(n, boundaries=[0, 0, 0, 0]):
    """
    Initialize the solution matrix with the boundary values.
    Args:
        n: number of grid points in each direction
        boundaries: list of boundary values [top, bottom, left, right]
    Returns:
        solution: initialized solution matrix
    """
    mean_value = np.mean(boundaries)
    solution = mean_value * np.ones(
        (n + 2, n + 2)
    )  # n+2 to include the boundary values

    # Set the boundary values
    solution[0, :] = boundaries[0]  # top
    solution[-1, :] = boundaries[1]  # bottom
    solution[:, 0] = boundaries[2]  # left
    solution[:, -1] = boundaries[3]  # right

    return solution


def jacobi(solution, n_iters=1):
    n = solution.shape[0]

    for _ in range(n_iters):
        solution_new = np.copy(solution)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                solution_new[i, j] = 0.25 * (
                    solution[i - 1, j]
                    + solution[i + 1, j]
                    + solution[i, j - 1]
                    + solution[i, j + 1]
                )
        solution = solution_new

    return solution_new


def gauss_seidel(solution, n_iters=1):
    n = solution.shape[0]

    for _ in range(n_iters):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                solution[i, j] = 0.25 * (
                    solution[i - 1, j]
                    + solution[i + 1, j]
                    + solution[i, j - 1]
                    + solution[i, j + 1]
                )

    return solution
