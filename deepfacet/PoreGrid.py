import os
import sys
import numpy as np
import subprocess
from molecular_builder import create_bulk_crystal, carve_geometry, write
from mlb_DodecahedronGeometry import *
from ase import Atom
from ovito.io import import_file, export_file


class PoreGrid:
    """
    Class for creating pore geometries of SiC for use in moledular dynamics
    simulations, relies heavily on the molecular-builder package:
    https://github.com/henriasv/molecular-builder
    """

    def __init__(self, cell_size, pore_radius, grid_shape):
        """
        Define size properties of a single cell and the total system size

        Params:
        * cell_size:      size of cell box in Angstrom
        * pore_radius:    (shortest) length from pore center to sides
        * shape:          system shape in cell units
        """
        self.cell_size = cell_size
        self.pore_radius = pore_radius
        self.grid_shape = grid_shape
        self.atoms = None

    def _create_atoms(self, geometries):
        bulk_atoms = create_bulk_crystal("silicon_carbide_3c", np.array(self.grid_shape)*self.cell_size)
        carved_atoms = None
        for geom in geometries:
            carve_geometry(bulk_atoms, geom, side="in")

        self.atoms = bulk_atoms

    def create_from_grid(self, grid):
        """
        Creates pores in a regular grid from a boolean array

        Params:
        * grid:    3d array, an element 1 corresponds to a pore which is present
        """

        condition = lambda x: x == 1
        geometries = self._get_geometries_regular(grid, condition)
        self._create_atoms(geometries)

    def _get_geometries_regular(self, grid, condition):
        q, r, s = self.grid_shape
        geometries = []
        self.pore_centers = []
        self.num_pores = 0
        for i in range(q):
            for j in range(r):
                for k in range(s):
                    if condition(grid[i, j, k]):
                        x = (i + 0.5)*self.cell_size
                        y = (j + 0.5)*self.cell_size
                        z = (k + 0.5)*self.cell_size
                        print(x, y, z)
                        self.pore_centers.append([x, y, z])
                        self.num_pores += 1
                        # dodeca_geom = DodecahedronGeometry(self.pore_radius, [x, y, z])
                        dodeca_geom = DodecahedronGeometry(self.pore_radius, [x, y, z])
                        geometries.append(dodeca_geom)

        return geometries

    def create_random_regular(self, p):
        """
        Randomly creates pores in a regular grid

        Params:
        * p:        probability of present pore at a site
        """
        grid = np.random.rand(*self.grid_shape)
        condition = lambda x: x <= p
        geometries = self._get_geometries_regular(grid, condition)
        self._create_atoms(geometries)

    def _ovito_write(self, filename):
        tmp_dir = "tmp"
        randint = np.random.randint(1, 10000)
        tmpfile = f"tmp{randint}.data"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.atoms.write(os.path.join(tmp_dir, tmpfile), format="lammps-data")

        pipeline = import_file(os.path.join(tmp_dir, tmpfile))
        os.remove(os.path.join(tmp_dir, tmpfile))
        export_file(pipeline, filename, "lammps/data", atom_style="atomic")

    def write_data(self, filename):
        """
        Write atom positions to file
        """
        assert (self.atoms is not None), "Grid must be created before write.."
        self._place_missing_atoms()
        self._ovito_write(filename)

        #insert pore numbers in atomdata file
        config_nums = "1 a # Present pores = ["
        for n in write_config_nums:
            config_nums += f"{n:.0f}, "
        config_nums = config_nums[:-2]
        config_nums += r"]\n"

        subprocess.call(["sed", "-i", config_nums, filename])

    def create_from_points(self, coords):
        """
        Converts coodrinates to a grid

        Params:
        * coords:   list of tuples with coordinates of the pores
        """

        grid = np.zeros(self.grid_shape)
        for c in coords:
            grid[c] = 1

        condition = lambda x: x == 1
        geometries = self._get_geometries_regular(grid, condition)
        self._create_atoms(geometries)

    def _place_missing_atoms(self):
        if self.num_pores == 0:
            return

        atomic_numbers = self.atoms.get_atomic_numbers()
        periodic_type, counts = np.unique(atomic_numbers, return_counts=True)
        atom_counts = dict(zip(periodic_type, counts))

        if atom_counts[6] > atom_counts[14]:
            missing_type = 'Si'
        elif atom_counts[6] < atom_counts[14]:
            missing_type = 'C'
        else:
            return

        diff = abs(atom_counts[6] - atom_counts[14])
        insert_positions = []
        np.random.shuffle(self.pore_centers)
        if self.num_pores < diff:

            offset = self.pore_radius/2.0

            for i in range(diff):
                idx = i%self.num_pores
                pos = np.copy(self.pore_centers[idx])
                if i < self.num_pores:
                    pos[0] += offset
                elif i < 2*self.num_pores:
                    pos[0] -= offset
                elif i < 3*self.num_pores:
                    # _ = 1 #do nothing (place in center)
                    pass

                elif i < 4*self.num_pores:
                    pos[1] += offset
                elif i < 5*self.num_pores:
                    pos[1] -= offset
                elif i < 6*self.num_pores:
                    pos[2] += offset
                elif i < 7*self.num_pores:
                    pos[2] -= offset
                else:
                    print("WARNING! missing atom/pore ratio too high..")
                    continue

                insert_positions.append(pos)
        else:
            insert_positions = np.copy(self.pore_centers[:diff])
        for pos in insert_positions:
            self.atoms.append(Atom(symbol=missing_type, position=pos))


if __name__ == '__main__':
    c = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

    grid = PoreGrid(50, 15, (3,3,3))
    grid.create_from_points(c)
    grid.create_random_regular(p=0.4)
    grid._ovito_write("random.data")






#
