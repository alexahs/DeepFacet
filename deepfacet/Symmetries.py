import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
import time, os, subprocess


class Symmetries:

    def __init__(self, n_pores, root_dir="data/config_files", config_criteria=None, include_syms=False):
        """
        Class for creating pore configurations in a 3^3 binary system,
        with or without symmetrical ones removed.
        Configurations are written to file for later use.
        Keeps track of configurations which have have been simulated.

        Params:
        * n_pores: number of pores
        * root_dir: directory to write the configurations
        * config criteria: used in InverseDesign.py, differentiates between search types
        """

        self.n_pores = n_pores
        self.root_dir = root_dir
        self.include_syms = include_syms

        if not include_syms:
            self.prev_gen_file = os.path.join(self.root_dir,f"generated_systems_{self.n_pores:.0f}.txt")
        else:
            self.prev_gen_file = None
        if config_criteria is not None:
            self.config_criteria_file = os.path.join(self.root_dir,f"generated_systems_{self.n_pores:.0f}_{config_criteria}.txt")
        else:
            self.config_criteria_file = None

        self.files_loaded = False

        return

    @staticmethod
    def print_map():
        """
        stdout mapping from num to grid coord
        """
        n = range(3)
        count = 1
        for k in n:
            for j in n:
                for i in n:
                    print(f"{count}:({i},{j},{k})")
                    count += 1

    @staticmethod
    def num_to_coord(num=None):
        """
        Mapping from grid number to grid coordinate (numbering as in print_num_to_coords)
        """
        coords = {}
        count = 1
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    coords[count] = (x, y, z)
                    count += 1

        if num is None:
            return coords
        else:
            return coords[num]

    @staticmethod
    def coord_to_num(coord=None):
        """
        Mapping from grid coordinate to grid number
        """
        nums = {}
        count = 1
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    nums[(x, y, z)] = count
                    count += 1

        if coord is None:
            return nums
        else:
            return nums[coord]

    @staticmethod
    def nums_to_grid(nums, num_map):
        """
        converts a sequence of pore numbers to a 3^3 grid
        """
        grid = np.zeros((3,3,3), dtype=np.uint8)

        for num in nums:
            coord = num_map[num]
            grid[coord] = 1

        return grid

    @staticmethod
    def shift_periodic(direction, p):
        """
        Converts a point to its periodic equivalent in a given direction
        Params:
        * direction:    3-tuple of direction to move to, +1/-1/0
        * p:            3-tuple of current point
        """
        i, j, k = direction
        n = 3
        new_coords = ((p[0] + i) % n, (p[1] + j) % n, (p[2] + k) % n)
        return new_coords

    @staticmethod
    def rotation_maps():
        mapping = {90: {}, 180: {}, 270:{}}
        for z in range(3):
            #cell IDs in current (x,y)-plane
            xy_plane_IDs = np.flip(np.arange((z+1)*9, z*9, -1).np.reshape(3, 3), axis=1)
            for k in range(1, 4):
                #rotate IDs k*90 degrees
                z_rotated = np.rot90(xy_plane_IDs, k=k)
                for ID, rot in zip(np.ravel(xy_plane_IDs), np.ravel(z_rotated)):
                    mapping[k*90][ID] = rot

        return mapping


    @staticmethod
    def flip_map():
        """rotate grid-cell IDs in the (x,z) and (y,z) planes"""

        # 180 degree rotations around x-axis
        dx_z0 = {1:21, 2:20, 3:19, 4:24, 5:23, 6:22, 7:27, 8:26, 9:25}
        dx_z1 = {10:12, 11:11, 12:10, 13:15, 14:14, 15:13, 16:18, 17:17, 18:16}
        dx_z2 = {val:key for key, val in dx_z0.items()} #dx_z2 = inverse of dx_z0

        # 180 degree rotations around y-axis
        dy_z0 = {1:25, 2:26, 3:27, 4:22, 5:23, 6:24, 7:19, 8:20, 9:21}
        dy_z1 = {10:16, 11:17, 12:18, 13:13, 14:14, 15:15, 16:10, 17:11, 18:12}
        dy_z2 = {val:key for key, val in dy_z0.items()} #dy_z2 = inverse of dy_z0

        mapping = {'x': {**dx_z0, **dx_z1, **dx_z2}, 'y': {**dy_z0, **dy_z1, **dy_z2}}

        return mapping

    @staticmethod
    def nums_to_n_hot(nums):
        n_hot = np.zeros(27, dtype=np.uint8)
        nums = list(nums)
        for idx in nums:
            idx = int(idx)
            n_hot[idx-1] = 1
        return n_hot

    @staticmethod
    def n_hot_to_nums(n_hot):
        nums = tuple(np.where(n_hot == 1)[0]+1)
        return nums

    def write_full_configuration_space(self):
        grid_nums = list(range(1, 28))
        all_conbination_nums = list(it.combinations(grid_nums, self.n_pores))
        header = f"# Full space of pore configurations of the 3^3 system with {self.n_pores:.0f} pores.\n"\
                  "#\n"\
                  "# Legend:\n"\
                  "#  z=0        z=1         z=2\n"\
                  "# 7 8 9    16 17 18    25 26 27\n"\
                  "# 4 5 6    13 14 15    22 23 24\n"\
                  "# 1 2 3    10 11 12    19 20 21\n"\
                  "#\n"\
                  "# y\n"\
                  "# ^>x\n"\
                  "# \n"\
                  "# \n"\
                 f"# Number of configurations: {len(all_conbination_nums)}\n\n\n"
        outpath = os.path.join(self.root_dir, f"all_pore_configs_{self.n_pores:.0f}.txt")
        np.savetxt(outpath, all_conbination_nums, fmt="%.0f", delimiter=",", header=header, comments="")

    def generate_configuration_library(self):
        """
        generates the possible coordinate combinations for a given pore number n
        with periodic and symmetric images removed

        Returns:
        * set of coordinate numbers
        """
        t0 = time.time()

        all_configurations = set()
        removed_configurations = set()
        removed_rotated = set()
        grid_nums = list(range(1, 28))
        num_to_coord_map = self.num_to_coord()
        coord_to_num_map = self.coord_to_num()
        number_combinations = it.combinations(grid_nums, self.n_pores)

        for nums in number_combinations:
            all_configurations.add(nums)

        idx = [-1, 0, 1]
        periodic_directions = [(x, y, z) for x in idx for y in idx for z in idx if not (x==y==z==0)]

        rotate_pore_nums = self.rotation_maps()
        flip_pore_nums = self.flip_map()


        #remove periodic and symmetric images
        for pore_nums in tqdm(all_configurations):

            #do nothing if this configuration has been checked
            if pore_nums in removed_configurations:
                continue

            #get coordinates from numbers
            orig_coords = []
            for num in pore_nums:
                orig_coords.append(num_to_coord_map[num])

            #find periodic images of this configuration
            for direction in periodic_directions:
                shifted_coords = []
                shifted_nums = []

                #shift coordinates
                for coord in orig_coords:
                    shifted_coords.append(self.shift_periodic(direction, coord))

                #convert shifted numbers to coordinates
                for coord in shifted_coords:
                    shifted_nums.append(coord_to_num_map[coord])

                #sort the numbers (all_configurations[i] will allways be sorted)
                shifted_nums.sort()
                shifted_nums = tuple(shifted_nums)

                # move to next configuration if image has already been checked
                if shifted_nums not in removed_configurations:
                    removed_configurations.add(shifted_nums)

            #find rotational symmetric images of this configuration
            for deg in (0, 90, 180, 270):
                if deg == 0:
                    rotated_nums = pore_nums
                else:
                    rotated_nums = []
                    for num in pore_nums:
                        rotated_nums.append(rotate_pore_nums[deg][num])

                    rotated_nums.sort()
                    rotated_nums = tuple(rotated_nums)

                    if rotated_nums not in removed_configurations:
                        removed_configurations.add(rotated_nums)

                # find flip symmetric images of this config
                for axis in ('x', 'y'):
                    flipped_nums = []
                    for num in rotated_nums:
                        flipped_nums.append(flip_pore_nums[axis][num])

                    flipped_nums.sort()
                    flipped_nums = tuple(flipped_nums)

                    if flipped_nums not in removed_configurations:
                        removed_configurations.add(flipped_nums)
        self.unique_configs = all_configurations.difference(removed_configurations)
        return

    def _generate_symmetric(self, out_dir, alternate_in_dir = None, max_syms = 33):
        """
        DEPRECATED

        Generates symmetric images of configurations from a file
        of pore configurations.

        Returns:
        With r_i a the representative configuration element
        and s_ij as a symmetric configuration element, they are
        placed in a (m+1, n, k) matrix, where n is the number
        of pores, m the max number of symmetric images (33 for 9 mores and
        colums = 0 if < m), k is the number of representatives,
        i.e each image has it's own (m+1, n) matrix with the representative
        along the first row and the symmetric images along the consecutive rows.

        k=0:
        _____________________________
        [r0  r1  r2  ...         rn |
        [s00 s01 s02 ...         s0n|
        |s10 s11 s12 ...         s1n|
        | .   .   .               . |
        | .   .   .               . |
        |sm0 sm1 sm2 ...         smn|
        _____________________________

        Each entry r_i and s_ij are represented by their coordinate
        numbers, eg. [1, 3, 4, 6, 9, 15, 16, 21, 23] for a 9 pore system.
        """

        pass

    @staticmethod
    def generate_rotational_symmetric(configs, periodic=False):
        """
        input:
            configs: array[n_samples, 27] of pore configs

        returns:
            dict[representative:tuple]->array[num_syms, 27]
        """

        nums = []
        for i in range(configs.shape[0]):
            nums.append(Three.n_hot_to_nums(configs[i,:]))

        num_to_coord_map = Three.num_to_coord()
        coord_to_num_map = Three.coord_to_num()
        rotate_pore_nums = Three.rotation_maps()
        flip_pore_nums = Three.flip_map()

        if periodic:
            idx = [-1, 0, 1]
            periodic_directions = [(x, y, z) for x in idx for y in idx for z in idx if not (x==y==z==0)]

        sym_dict = {}
        for pore_nums in nums:

            count = 0
            added_symmetries = set()
            #find rotational symmetric images of this configuration
            for deg in (0, 90, 180, 270):
                if deg == 0:
                    rotated_nums = pore_nums
                else:
                    rotated_nums = []
                    for num in pore_nums:
                        rotated_nums.append(rotate_pore_nums[deg][num])

                    rotated_nums.sort()
                    rotated_nums = tuple(rotated_nums)

                    if rotated_nums not in added_symmetries:
                        count += 1
                        added_symmetries.add(rotated_nums)
                # find flip symmetric images of this config
                for axis in ('x', 'y'):
                    flipped_nums = []
                    for num in rotated_nums:
                        flipped_nums.append(flip_pore_nums[axis][num])

                    flipped_nums.sort()
                    flipped_nums = tuple(flipped_nums)

                    if flipped_nums not in added_symmetries:
                        count += 1
                        added_symmetries.add(flipped_nums)

            #get coordinates from numbers
            orig_coords = []
            for num in pore_nums:
                orig_coords.append(num_to_coord_map[num])

            #find periodic images of this configuration
            if periodic:
                for direction in periodic_directions:
                    shifted_coords = []
                    shifted_nums = []

                    #shift coordinates
                    for coord in orig_coords:
                        shifted_coords.append(Three.shift_periodic(direction, coord))

                    #convert shifted numbers to coordinates
                    for coord in shifted_coords:
                        shifted_nums.append(coord_to_num_map[coord])

                    #sort the numbers (all_configurations[i] will allways be sorted)
                    shifted_nums.sort()
                    shifted_nums = tuple(shifted_nums)

                    # move to next configuration if image has already been checked
                    if (shifted_nums not in added_symmetries):# and (len(added_symmetries) < 28):
                        count += 1
                        added_symmetries.add(shifted_nums)


            symmetries = np.array(list(added_symmetries))
            one_hot = np.zeros((count, 27), dtype=np.uint8)
            for i, nums in enumerate(symmetries):
                one_hot[i,:] = Three.nums_to_n_hot(nums)

            repr_bool = tuple(Three.nums_to_n_hot(pore_nums))

            sym_dict[repr_bool] = one_hot

        return sym_dict

    def write_configurations(self, ignore_warning=False):
        if self.n_pores > 13:
            #requires inverse pore config file to exist
            self.unique_configs = set()
            grid_nums = set(range(1, 28))
            inverse_num_pores = 27 - self.n_pores
            inverse_configs = self.from_file(inverse_num_pores)
            for config in inverse_configs.values():
                new_config = grid_nums.difference(config)
                self.unique_configs.add(tuple(new_config))

        assert(self.root_dir[0] != "/"), "invalid file destination"
        header = f"# Distinct pore configurations of the 3^3 system with {self.n_pores:.0f} pores.\n"\
                  "#\n"\
                  "# Legend:\n"\
                  "#  z=0        z=1         z=2\n"\
                  "# 7 8 9    16 17 18    25 26 27\n"\
                  "# 4 5 6    13 14 15    22 23 24\n"\
                  "# 1 2 3    10 11 12    19 20 21\n"\
                  "#\n"\
                  "# y\n"\
                  "# ^>x\n"\
                  "# \n"\
                  "# \n"\
                 f"# Number of configurations: {len(self.unique_configs)}\n\n\n"

        outpath = os.path.join(self.root_dir, f"pore_configs_{self.n_pores}.txt")

        if not ignore_warning:
            if os.path.exists(outpath):
                print(f"Warning: {outpath} already exists.. ")
                user_input = input(f"Overwrite? [Y/n]")
                if user_input == "Y":
                    subprocess.call(["rm", outpath])
                else:
                    print("Writing terminated")
                    return
        np.savetxt(outpath, list(self.unique_configs), fmt="%.0f", delimiter=",", header=header, comments="")

    @staticmethod
    def get_pore_coords(nums):
        """
        returns a list of coordinates for the privided list of pore numbers
        """

        pore_map = Three.num_to_coord()
        coords = []
        for num in nums:
            coords.append(pore_map[num])

        return coords

    def _from_file(self, n_pores, alternate_dir = None):
        """
        Loads all pore configurations from file and returns the pore numbers as dict
        """

        if self.include_syms:
            print("Loading all possible configs from file, will take some time..")
            fname = os.path.join(self.root_dir, f"all_pore_configs_{n_pores}.txt")
        else:
            fname = os.path.join(self.root_dir, f"pore_configs_{n_pores}.txt")
        if alternate_dir is not None:
            fname = alternate_dir

        # if alternate_dir is None:
        #     fname = os.path.join(self.root_dir, f"pore_configs_{n_pores}.txt")
        # else:
        #     fname = alternate_dir
        assert(os.path.exists(fname)), f"coord file {fname} does not exist"
        data = np.genfromtxt(fname, dtype=np.uint8, skip_header=16, delimiter=',')
        d = {}
        print("Converting formats..")
        for i in tqdm(range(data.shape[0])):
            d[i+1] = tuple(data[i])
        return d

    def load_config_files(self):

        header = "# List of 3^3 pore systems that have been generated\n"\
                 "# Format: 1st column is the number of pores, 2nd column is the configuration number,\n"\
                 "# which corresponds to the line number (minus 16) in pore_configs_xx.txt\n\n"\
                 "0, 0\n"\
                 "0, 0\n"\

        # print(self.prev_gen_file)
        # print(self.config_criteria_file)

        if not self.include_syms:
            if not os.path.exists(self.prev_gen_file):
                with open(self.prev_gen_file, "w") as outfile:
                    outfile.write(header)

        if not os.path.exists(self.config_criteria_file) and self.config_criteria_file is not None:
           with open(self.config_criteria_file, "w") as outfile:
               outfile.write(header)

        if not self.files_loaded:
            self.coord_map = self._from_file(self.n_pores)
            self.prev_generated = set()
            if not self.include_syms:
                tmp = np.genfromtxt(self.prev_gen_file, delimiter=',', dtype=np.int, skip_header=4)
                for pores, config in tmp:
                    self.prev_generated.add(config)

            if self.config_criteria_file is not None:
                tmp2 = np.genfromtxt(self.config_criteria_file, delimiter=',', dtype=np.int, skip_header=4)
                for pores, config in tmp2:
                    self.prev_generated.add(config)

            self.num_configs = len(self.coord_map)
            self.files_loaded = True

    def filter_custom_configs(self, pore_nums:list):
        """
        Filters the provided pore numbers through ones contained in the previously
        simulated systems file. Returns dict of config_num -> pore_nums

        Adds the provided pore numbers to file containing previously simulated systems
        and removes from the list ones which are prev generated.
        Returns the filtered list.

        pore_nums: list of tuples containing pore numbers
        """

        self.load_config_files()
        coord_map_inverse = {}
        filtered_pore_nums = {}

        for key, val in self.coord_map.items():
            coord_map_inverse[val] = key

        for pore_num in pore_nums:
            config_num = coord_map_inverse[pore_num]
            if config_num not in self.prev_generated:
                filtered_pore_nums[config_num] = pore_num
            else:
                print(f"pore_nums {pore_num} already generated. Skipping..")

        if self.config_criteria_file is not None:
            out_fname = self.config_criteria_file
        else:
            out_fname = self.prev_gen_file

        with open(out_fname, "a") as outfile:
            for pore_num in filtered_pore_nums.values():
                config_num = coord_map_inverse[pore_num]
                outfile.write(f"{self.n_pores}, {config_num}\n")

        print(f"Filtered : {len(filtered_pore_nums)}/{len(pore_nums)} configs")
        return filtered_pore_nums

    def pick_random(self, max_attempts = 1000):
        """
        chooses randomly a configuration

        returns: configuration number:int, pore numbers: list of 3-tuples
        """

        self.load_config_files()


        for attempt in range(max_attempts):
            candidate = np.random.randint(1, self.num_configs)
            if candidate in self.prev_generated:
                continue
            else:
                with open(self.prev_gen_file, "a") as outfile:
                    outfile.write(f"{self.n_pores}, {candidate}\n")
                return candidate, self.coord_map[candidate]

            attempt +=1
        else:
            print(f"Failed picking random config not run ({max_attempts} attempts)")

        return None


if __name__ == '__main__':
    pass























#
