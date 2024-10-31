# Monte Carlo method for solving PDE
def monte_carlo_pde_solution(n, r, sig, T, N_simulation, boundaries):
    np.random.seed(seed=42)
    solution = np.zeros((n + 2, n + 2))
    mean_value = np.mean(boundaries)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            W = (r - sig**2 / 2) * T + ss.norm.rvs(
                loc=0, scale=sig, size=N_simulation
            )
            avg_value = np.mean(mean_value * np.exp(W))
            solution[i, j] = np.exp(-r * T) * avg_value

    return solution
