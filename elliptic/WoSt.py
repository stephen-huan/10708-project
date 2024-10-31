import numpy as np
import random

# Function to calculate the distance from a point to the Dirichlet boundary
def distance_to_dirichlet(x, boundary_dirichlet):
    min_distance = float('inf')
    for polyline in boundary_dirichlet:
        for i in range(len(polyline) - 1):
            a, b = polyline[i], polyline[i + 1]
            u = b - a
            t = np.clip(np.dot(x - a, u) / np.dot(u, u), 0.0, 1.0)
            y = (1.0 - t) * a + t * b
            min_distance = min(min_distance, np.linalg.norm(x - y))
    return min_distance

# Function to calculate the distance to the closest silhouette point on the Neumann boundary
def distance_to_silhouette(x, boundary_neumann):
    min_distance = float('inf')
    for polyline in boundary_neumann:
        for i in range(1, len(polyline) - 1):
            if is_silhouette(x, polyline[i - 1], polyline[i], polyline[i + 1]):
                min_distance = min(min_distance, np.linalg.norm(x - polyline[i]))
    return min_distance

# Function to determine if a point is a silhouette point
def is_silhouette(x, a, b, c):
    cross1 = np.cross(b - a, x - a)
    cross2 = np.cross(c - b, x - b)
    return cross1 * cross2 < 0

# Function to perform a ray intersection with a boundary or a star-shaped region
def ray_intersection(x, v, r, boundary_neumann):
    min_t = r
    for polyline in boundary_neumann:
        for i in range(len(polyline) - 1):
            a, b = polyline[i], polyline[i + 1]
            u = b - a
            w = x - a
            d = np.cross(v, u)
            if d != 0:
                s = np.cross(v, w) / d
                t = np.cross(u, w) / d
                if t > 0 and 0 <= s <= 1:
                    min_t = min(min_t, t)
    return x + min_t * v

# Function to solve the PDE using Walk on Stars (WoSt) algorithm
def walk_on_stars(x0, boundary_dirichlet, boundary_neumann, g, eps=0.0001, r_min=0.0001, n_walks=10000, max_steps=10000):
    sum_boundary_values = 0.0
    for _ in range(n_walks):
        x = np.array(x0)
        on_boundary = False
        steps = 0
        while steps < max_steps:
            # Compute radius for star-shaped region
            d_dirichlet = distance_to_dirichlet(x, boundary_dirichlet)
            d_silhouette = distance_to_silhouette(x, boundary_neumann)
            r = max(r_min, min(d_dirichlet, d_silhouette))
            
            # Sample random direction
            theta = random.uniform(0, 2 * np.pi)
            if on_boundary:
                theta /= 2
            v = np.array([np.cos(theta), np.sin(theta)])
            
            # Find the next point using ray intersection
            x = ray_intersection(x, v, r, boundary_neumann)
            
            # Check if the walk reached the Dirichlet boundary
            if d_dirichlet <= eps:
                break
            
            steps += 1
        
        sum_boundary_values += g(x)
    
    return sum_boundary_values / n_walks

# Example usage
if __name__ == "__main__":
    # Define boundaries
    boundary_dirichlet = [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
    boundary_neumann = [np.array([[0.5, 1], [0.5, 1.5], [1, 1.5], [1, 1]])]
    
    # Function to evaluate Dirichlet boundary condition
    def g(x):
        return np.linalg.norm(x)  # Example function
    
    # Initial point
    x0 = np.array([0.2, 0.2])
    
    # Solve PDE using Walk on Stars
    result = walk_on_stars(x0, boundary_dirichlet, boundary_neumann, g)
    print("Estimated solution at x0:", result)
