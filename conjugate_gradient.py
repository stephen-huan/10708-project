import numpy as np

def conjugate_gradient(A, b, x, tol, max_iter):
    r = b-A@x
    if np.linalg.norm(r) < tol: return x
    p = r.copy()
    r_old = np.transpose(r) @ r

    for k in range(max_iter):
        alpha = r_old / (np.transpose(p)@(A@p))
        x = x + alpha*p
        r = r- alpha*(A@p)

        if np.linalg.norm(r) < tol: return x

        r_new = np.transpose(r)@r
        beta = (r_new)/ (r_old)
        p = r + beta*p
        
        r_old =  r_new
    return x