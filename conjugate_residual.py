import numpy as np

def conjugate_residual(A, x0, b, max_iter, tol):
    x = x0.copy()
    r = b-A@x0
    p = r.copy()
    Ap = A@p

    for k in range(max_iter):
        Ar = A@r
        alpha = (np.transpose(r) @ Ar) /  (np.transpose(Ap)@Ap)
        x_new = x + alpha*p

       # Check for convergence based on the change in x
        if np.linalg.norm(x_new - x) / np.linalg.norm(x_new) < tol:
            return x_new

        r_new = r - alpha * Ap
        Ar_new = A@r_new
        beta = (np.transpose(r_new) @ Ar_new) / (np.transpose(r) @ Ar)
        p_new = r_new + beta * p
        Ap_new = Ar_new + beta * Ap

        x, r, p, Ap = x_new, r_new, p_new, Ap_new
    return x