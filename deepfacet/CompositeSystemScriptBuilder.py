import os, sys, subprocess
import numpy as np
from PoreGrid import PoreGrid
from Symmetries import import Symmetries
from tqdm import tqdm
from pathlib import Path

class System:
    def __init__(self, time, strain_rate, temperature, config_name, root_dir, run_template):
        """
        [time] = ns
        [strain_rate] = 1/ns
        [temperature] = K
        [config_name] = (num_pores, config_num) as defined in "pore_configs_x.txt"
        """


        self.time = time
        self.strain_rate = strain_rate
        self.temperature = temperature
        self.config_name = config_name
        self.root_dir = root_dir
        self.run_template = run_template
        self.template_dir = "script_templates/"

    @property
    def num_timesteps(self):
        dt = 0.002
        n_steps = int(self.time/dt*1000.0)
        return n_steps

    def build_path(self):
        sim_dir= f"pore_{self.config_name}_T{self.temperature}_t{self.time}_sr{self.strain_rate}"
        wd = os.path.join(self.root_dir, sim_dir)
        if os.path.exists(wd):
            print(f"Warning! {wd} already exists")
            answer = input("overwrite? [YES/no]\n")
            if answer == "YES":
                subprocess.call(f"rm -rf {wd}".split())
                os.makedirs(wd)
            else:
                print("Build terminated..")
                exit()
        else:
            os.makedirs(wd)

        self.wd = wd
        return wd

    def process_scripts(self):
        wd = self.build_path()


        with open(os.path.join(self.template_dir,self.run_template), "r") as infile:
            run_script = infile.readlines()


        idx_num_steps = None
        idx_temp = None
        idx_erate = None

        i = 0
        for line in run_script:
            if "# --time def" in line:
                idx_num_steps = i+1
            if "# -- temp def" in line:
                idx_temp = i+1
            if "# --deform def" in line:
                idx_erate = i+1
            i+=1

        assert(idx_num_steps != None)   ,"num steps not in template"
        assert(idx_temp != None)        ,"temp not in template"
        assert(idx_erate != None)       ,"erate not in template"


        run_script[idx_num_steps] = f"variable TIME_STEPS\t\t equal {self.num_timesteps} #{self.time} ns \n"
        run_script[idx_temp] = f"variable T\t\t\t equal {self.temperature} \n"
        run_script[idx_erate] = f"variable ERATE\t equal {-self.strain_rate*0.001} \n"

        with open(os.path.join(wd, "run.lmp"), "w") as outfile:
            outfile.writelines(run_script)


        with open(os.path.join(self.template_dir, "job_template.sh"), "r") as infile:
            job_script = infile.readlines()

        job_idx = None

        i = 0
        for line in job_script:
            if "#SBATCH --job-name" in line:
                job_idx = i
            i += 1

        assert(job_idx != None), "job name not in template"
        job_script[job_idx] = f"#SBATCH --job-name={self.config_name} \n"

        with open(os.path.join(wd, "job.sh"), "w") as outfile:
            outfile.writelines(job_script)


        subprocess.call(f"cp {self.template_dir}/SiC.vashishta {wd}".split())

class RerunSystem(System):
    def __init__(self, time, strain_rate, temperature, config_name, root_dir, run_template, rerun_template, num_reruns=4):
        super().__init__(time, strain_rate, temperature, config_name, root_dir, run_template)
        self.rerun_template = rerun_template
        self.num_reruns = num_reruns

    def process_scripts(self):
        wd = self.build_path()

        """
        ===============
        Main script
        ===============
        """
        with open(os.path.join(self.template_dir,self.run_template), "r") as infile:
            run_script = infile.readlines()

        idx_num_steps = None
        idx_temp = None
        idx_erate = None

        i = 0
        for line in run_script:
            if "# --time def" in line:
                idx_num_steps = i+1
            if "# -- temp def" in line:
                idx_temp = i+1
            if "# --deform def" in line:
                idx_erate = i+1
            i+=1

        assert(idx_num_steps != None)   ,"num steps not in template"
        assert(idx_temp != None)        ,"temp not in template"
        assert(idx_erate != None)       ,"erate not in template"

        run_script[idx_num_steps] = f"variable TIME_STEPS\t\t equal {self.num_timesteps} #{self.time} ns \n"
        run_script[idx_temp] = f"variable T\t\t\t equal {self.temperature} \n"
        run_script[idx_erate] = f"variable ERATE\t equal {-self.strain_rate*0.001} \n"

        with open(os.path.join(wd, "run.lmp"), "w") as outfile:
            outfile.writelines(run_script)


        with open(os.path.join(self.template_dir, "job_template_multirun.sh"), "r") as infile:
            job_script = infile.readlines()

        job_idx = None
        rerun_idx = None
        i = 0
        for line in job_script:
            if "#SBATCH --job-name" in line:
                job_idx = i
            if "#NUM_RERUNS DEF" in line:
                rerun_idx = i+1
            i += 1

        assert(job_idx != None), "job name not in job template"
        assert(rerun_idx != None), "rerun def not in job template"
        job_script[job_idx] = f"#SBATCH --job-name={self.config_name} \n"
        job_script[rerun_idx] = f"num_reruns={self.num_reruns} \n"

        with open(os.path.join(wd, "job.sh"), "w") as outfile:
            outfile.writelines(job_script)
        subprocess.call(f"cp {self.template_dir}/SiC.vashishta {wd}".split())


        """
        ===============
        Rerun scripts
        ===============
        """
        with open(os.path.join(self.template_dir, self.rerun_template), "r") as infile:
            rerun_script_get_idx = infile.readlines()

        k = 0
        seed_idx = None
        for line in rerun_script_get_idx:
            if "# -- seed def" in line:
                seed_idx = k+1
                break
            k+=1

        assert(seed_idx != None)   ,f"seed def not in {self.rerun_template}"


        seeds_used = []


        for i in range(self.num_reruns):
            rerun_dir = os.path.join(wd, f"rerun_{i+1}")
            os.makedirs(rerun_dir)
            source = os.path.join(self.template_dir, self.rerun_template)
            target = os.path.join(rerun_dir, "run.lmp")
            subprocess.call(f"cp {source} {target}".split())

            with open(target, "r") as infile:
                rerun_script = infile.readlines()


            for rnd in np.random.randint(low=1, high=1000, size=100):
                if rnd not in seeds_used:
                    seed = rnd
                    seeds_used.append(rnd)
                    break


            rerun_script[seed_idx] = f"variable SEED \t\t equal {seed} \n"

            with open(os.path.join(target), "w") as outfile:
                outfile.writelines(rerun_script)


        assert(len(seeds_used) == self.num_reruns), "not all seeds are set"

        subprocess.call(f"cp {self.template_dir}/copy.py {wd}".split())


def generate_simple_run(n_systems, target_dir, time=10, strain_rate=0.006, temp=2200, configs_dir=None, shape=None, pore_radius=None, num_pores=None, cell_size=None, run_template=None):
    if configs_dir is None:
        configs_dir = "coord_files_threes_new"



    pore_picker = Symmetries(n_pores=num_pores, root_dir=configs_dir)
    coord_map = pore_picker.num_to_coord()
    for i in range(n_systems):
        config_number, pore_numbers = pore_picker.pick_random()
        pore_coords = pore_picker.get_pore_coords(pore_numbers)
        config_name = f"{num_pores}_{config_number}"

        system = System(time, strain_rate, temp, config_name, target_dir, run_template)
        system.process_scripts()

        grid = PoreGrid(cell_size, pore_radius, shape)
        grid.create_from_points(pore_coords)
        grid.write_data(filename=f"{system.wd}/atoms.data", write_config_nums=pore_numbers)

        print(f"Created {system.wd}")
        print(f"config: {config_name}\n")


def generate_multi_run(n_systems, target_dir, time=2.3, strain_rate=0.03, temp=2200, configs_dir=None, shape=None, pore_radius=None, num_pores=None, cell_size=None, run_template=None):
    if configs_dir is None:
        configs_dir = "coord_files_threes_new"

    rerun_template = "rerunThrees_2.3ns.lmp"
    num_reruns = 4

    pore_picker = Symmetries(n_pores=num_pores, root_dir=configs_dir)
    coord_map = pore_picker.num_to_coord()
    print("TARGET DIR", target_dir)
    for i in tqdm(range(n_systems)):
        config_number, pore_numbers = pore_picker.pick_random()
        pore_coords = pore_picker.get_pore_coords(pore_numbers)
        config_name = f"{num_pores}_{config_number}"


        system = RerunSystem(time, strain_rate, temp, config_name, target_dir, run_template, rerun_template=rerun_template, num_reruns=num_reruns)
        system.process_scripts()

        grid = PoreGrid(cell_size, pore_radius, shape)
        grid.create_from_points(pore_coords)
        grid.write_data(filename=f"{system.wd}/atoms.data", write_config_nums=pore_numbers)


def generate_screened_run(pore_nums, target_dir, time=2.3, strain_rate=0.03, temp=2200, configs_dir=None, shape=None, pore_radius=None, num_pores=None, cell_size=None, run_template=None, search_criteria=None):
    if configs_dir is None:
        configs_dir = "coord_files_threes_new"

    rerun_template = f"rerunThrees_{time:.1f}ns.lmp"
    num_reruns = 4

    if "with_syms" in search_criteria:
        include_syms = True
    else:
        include_syms = False

    pore_picker = Symmetries(n_pores=num_pores, root_dir=configs_dir, config_criteria=search_criteria, include_syms=include_syms)
    coord_map = pore_picker.num_to_coord()
    print("TARGET DIR", target_dir)

    pore_nums_dict = pore_picker.filter_custom_configs(pore_nums)
    print("Generating simulation files..")
    for config_number, pore_numbers in tqdm(pore_nums_dict.items()):

        pore_coords = pore_picker.get_pore_coords(pore_numbers)
        config_name = f"{num_pores}_{config_number}"

        system = RerunSystem(time, strain_rate, temp, config_name, target_dir, run_template, rerun_template=rerun_template, num_reruns=num_reruns)
        system.process_scripts()

        grid = PoreGrid(cell_size, pore_radius, shape)
        grid.create_from_points(pore_coords)
        grid.write_data(filename=f"{system.wd}/atoms.data", write_config_nums=pore_numbers)



def generate_custom_run(pore_nums, target_dir, time=2.3, strain_rate=0.03, temp=2200, configs_dir=None, shape=None, pore_radius=None, num_pores=None, cell_size=None, run_template=None):
    rerun_template = f"rerunThrees_2.8ns.lmp"
    config_name = "custom"
    num_reruns = 5

    system = RerunSystem(time, strain_rate, temp, config_name, target_dir, run_template, rerun_template=rerun_template, num_reruns=num_reruns)
    system.process_scripts()

    pore_coords = Symmetries.get_pore_coords(pore_nums)
    grid = PoreGrid(cell_size, pore_radius, shape)
    grid.create_from_points(pore_coords)
    grid.write_data(filename=f"{system.wd}/atoms.data", write_config_nums=pore_nums)


def get_base_config():
    """
    DO NOT CHANGE
    """
    DEFAULT_CELL_SIZE = 50
    DEFAULT_PORE_RADIUS = 15
    DEFAULT_NUM_PORES = 9
    DEFAULT_SHAPE = (3, 3, 3)
    DEFAULT_TEMP = 2200
    BASE_CONFIG = {
        "cell_size" : DEFAULT_CELL_SIZE,
        "pore_radius" : DEFAULT_PORE_RADIUS,
        "num_pores" : DEFAULT_NUM_PORES,
        "shape" : DEFAULT_SHAPE,
        "temp" : DEFAULT_TEMP,
    }
    return BASE_CONFIG

if __name__ == '__main__':

    pass






#
