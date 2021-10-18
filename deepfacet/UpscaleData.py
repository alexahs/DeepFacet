import numpy as np

class BoolToImage:
    """
    Tool for visualizing pores in the 3^3 grid of pores
    Creates 2d grids for each of the xy-planes of the grid
    """

    def __init__(self, n, r):
        if n % 2 != 1:
            n += 1
        self.n = n
        self.r = r
        self.c = n//2

    def draw(self):
        n = self.n
        img = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                distance = self.distance_from_center(i, j)
                if distance < self.r**2:
                    img[i, j] = 1 - distance/self.r**2
        return img

    def distance_from_center(self, i, j):
        d = (i - self.c)**2 + (j - self.c)**2
        d /= self.n**2
        return d

    def fill_grid(self, grid):
        n = self.n
        nx, ny, nz = grid.shape[0]*n, grid.shape[1]*n, 3
        processed_grid = np.zeros((nx, ny, nz))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    cell = grid[i, j, k]
                    if cell == 1:
                        img = self.draw()
                    else:
                        img = np.zeros((n, n))
                    processed_grid[i*n:(i+1)*n, j*n:(j+1)*n, k] = img
        return processed_grid
