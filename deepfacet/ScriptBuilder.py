import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from molecular_builder import create_bulk_crystal, carve_geometry, write
from molecular_builder.geometry import SphereGeometry, PlaneGeometry
from ovito.io import import_file, export_file
# sys.path.append(".")
import settings

class ScriptBuilder:
    """
    Base class for setting up simulation files
    """
    def __init__(self):
        self.job_template = settings.STANDARD_JOB_TEMPLATE
        self.dt = 0.002 #ps

    @property
    def timesteps(self):
        n_steps = self.t/self.dt*1000
        return int(n_steps)

    @staticmethod
    def _ovito_write(atoms, filename):
        tmp_dir = Path(settings.TMP_ATOMS_PATH)
        randint = np.random.randint(1, 10000)
        tmpfile = tmp_dir / f"tmp_{np.random.randint(1, 10000)}.data"
        if not tmp_dir.exists():
            tmp_dir.mkdir()
        atoms.write(tmpfile, format="lammps-data")

        pipeline = import_file(str(tmpfile))
        os.remove(tmpfile)
        export_file(pipeline, str(filename), "lammps/data", atom_style="atomic")

    def _build_simulation_path(self):
        if self.wd.exists():
            print(f"Warning: {self.wd} already exists")
            inp = input("overwrite? [Y/n]\n")
            if inp == "Y":
                subprocess.call(["rm", "-rf", self.wd])
                self.wd.mkdir()
            else:
                print("build terminated..")
                exit(1)
        else:
            self.wd.mkdir()

    def _write_job_script(self):
        with open(self.job_template, "r") as fp:
            job_script = fp.readlines()
        job_idx = None
        i = 0
        for line in job_script:
            if "#SBATCH --job-name" in line:
                job_idx = i
            i += 1
        assert(job_idx)
        job_script[job_idx] = f"#SBATCH --job-name={self.job_name}\n"

        with open(self.wd / "job.sh", "w") as fp:
            fp.writelines(job_script)

    def build(self):
        self._build_simulation_path()
        with open(self.lmp_template, "r") as fp:
            lmp_script = fp.readlines()

        self._write_job_script()

        with open(self.wd / "run.lmp", "w") as fp:
            fp.writelines(self._prepare_lmp_script(lmp_script))

        subprocess.call(f"cp {settings.FORCE_FIELD} {self.wd / 'SiC.vashishta'}".split())
        self._build_cont()

    def _prepare_lmp_script(self, lmp_template):
        return NotImplementedError

    def _build_cont(self):
        pass

class SphereFaceting(ScriptBuilder):
    """
    Class for creating simulations of an initially spherical pore
    Params:
    * n: system size [Å]
    * r: radius of sphere [Å]
    * t: simulation time [ns]
    """
    def __init__(self, n, r, t):
        super().__init__()
        self.n = n
        self.r = r
        self.t = t
        self.wd = Path(settings.FACETING_SIMS) / f"n{n}_r{r}_t{t}"
        self.lmp_template = Path(settings.FACETING_SCRIPT)
        self.job_name = "facet"


    def _prepare_lmp_script(self, lmp_template):
        idx_time = None
        i = 0
        for line in lmp_template:
            if "# --time def" in line:
                idx_time = i+1
            i+=1

        assert(idx_time)
        lmp_template[idx_time] = f"variable TIME_STEPS equal {self.timesteps} \n"
        return lmp_template

    def _build_cont(self):
        c = self.n//2
        atoms = create_bulk_crystal("silicon_carbide_3c", [self.n]*3)
        geometry = SphereGeometry((c, c, c), self.r)
        carve_geometry(atoms, geometry, side="in")
        self._ovito_write(atoms, self.wd / "atoms.data")

class DeformSingle(ScriptBuilder):
    """
    Builds simulation files for deforming a faceted pore
    Params:
    * T: system temperature [K]
    * t: simulation time [ns]
    * deform scale: final z-length of simulation box [percent]
    """
    def __init__(self, T, t, scale=0.85):
        super().__init__()
        self.T = T
        self.t = t
        self.scale = scale

        self.wd = Path(settings.DEFORM_SINGLE_SIMS) / f"T{T}_t{t}_scale{scale}"
        self.lmp_template = Path(settings.DEFORM_SINGLE_SCRIPT)
        self.job_name = "deform"

    def _prepare_lmp_script(self, lmp_template):
        idx_num_steps = None
        idx_temp = None
        idx_deform = None
        idx_atom_def = None
        idx_seed = None
        i = 0
        for line in lmp_template:
            if "# --time def" in line:
                idx_num_steps = i+1
            if "# --temp def" in line:
                idx_temp = i+1
            if "# --deform def" in line:
                idx_deform = i+1
            if "# --atom def" in line:
                idx_atom_def = i+1
            if "# --seed" in line:
                idx_seed = i+1

            i += 1

        assert(idx_num_steps)
        assert(idx_temp)
        assert(idx_deform)
        assert(idx_atom_def)
        assert(idx_seed)

        lmp_template[idx_num_steps] = f"variable TIME_STEPS equal {self.timesteps} \n"
        lmp_template[idx_temp] = f"variable T equal {self.T} \n"
        lmp_template[idx_deform] = "variable DEFORM_SCALE equal {self.deform_scale} \n"
        lmp_template[idx_atom_def] = f"read_restart inverted_crystal_T{self.T}.restart"

        return lmp_template

    def _build_cont(self):
        restart_file = Path(settings.DEFORM_SINGLE_RESTART) / f"inverted_crystal_T{self.T}.restart"
        if not restart_file.exists():
            print(f"restart file {restart_file} does not exist.")
            exit()

        cmd = f"cp {restart_file} {self.wd / f'inverted_crystal_T{self.T}.restart'}"
        subprocess.call(cmd.split())

class CreepInit(ScriptBuilder):
    """
    Creates initial condition (restart files) for creep simulations
    Params:
    * T: system temperature [K]
    * srate: strain rate [1/ns]
    * peak_prox: writes restart files when this close to yield strain [percent]
    """
    def __init__(self, T, srate, peak_prox=0.6):
        super().__init__()
        self.T = T
        self.srate = srate
        self.peak_prox = peak_prox
        self.wd = Path(settings.CREEP_INIT_SIMS) / f"T{T}"
        self.lmp_template = Path(settings.CREEP_INIT_SCRIPT)
        self.job_name = "creep_init"

    def _get_target_strain(self):
        bias = 0.02*1.5
        models = {
                -0.006: 0.17390398592201742-4.559999294021498e-05*self.T - bias
        }
        try:
            target_strain =  models[self.srate]
        except(KeyError):
            print(f"No model for erate={self.srate}. Possible strain rates: {models.keys()}")

        target_strain += self.peak_prox*bias
        return target_strain

    def _prepare_lmp_script(self, lmp_template):
        idx_temp = None
        idx_erate = None
        idx_target_strain = None
        i = 0
        for line in lmp_template:
            if "# --temp def" in line:
                idx_temp = i+1
            if "# --deform def" in line:
                idx_erate = i+1
            if "# --def target strain" in line:
                idx_target_strain = i+1

            i += 1

        assert(idx_temp)
        assert(idx_erate)
        assert(idx_target_strain)

        lmp_template[idx_temp]            = f"variable T              equal {self.T} \n"
        lmp_template[idx_erate]           = f"variable ERATE          equal {self.srate*0.001:.4e} \n"
        lmp_template[idx_target_strain]   = f"variable target_strain  equal {self._get_target_strain():.4f}"
        return lmp_template

    def _build_cont(self):
        creep_init_files_path = self.wd / "restart_files"
        if not creep_init_files_path.exists():
            creep_init_files_path.mkdir()
        restart_file = Path(settings.DEFORM_SINGLE_RESTART) / f"inverted_crystal_T{self.T}.restart"
        if not restart_file.exists():
            print(f"restart file {restart_file} does not exist.")
            exit()

        cmd = f"cp {restart_file} {self.wd / f'inverted_crystal_T{self.T}.restart'}"
        subprocess.call(cmd.split())

class RunCreep(ScriptBuilder):
    """
    Creep simulation
    Params:
    * T: system temperature [K]
    * t: simulation time [ns]
    * restart_id: restart file ID from sims generated by CreepInit class
    """
    def __init__(self, T, t, restart_id):
        super().__init__()
        self.T = T
        self.t = t
        self.restart_id = restart_id
        self.wd = Path(settings.CREEP_RUN_SIMS) / f"T{T}_run{restart_id}"
        self.lmp_template = Path(settings.CREEP_RUN_SCRIPT)
        self.job_name = "run_creep"


    def _prepare_lmp_script(self, lmp_template):

        idx_num_steps = None
        idx_temp = None
        i = 0
        for line in lmp_template:
            if "# --time def" in line:
                idx_num_steps = i+1
            if "# --temp def" in line:
                idx_temp = i+1
            i += 1

        assert(idx_num_steps)
        assert(idx_temp)

        lmp_template[idx_num_steps]       = f"variable TIME_STEPS equal {self.timesteps} \n"
        lmp_template[idx_temp]            = f"variable T              equal {self.T} \n"

        return lmp_template

    def _build_cont(self):
        creep_init_files_path = self.wd / "restart_files"
        if not creep_init_files_path.exists():
            creep_init_files_path.mkdir()
        restart_file = Path(settings.CREEP_INIT_SIMS) / f"T{self.T}/restart_files/restart_num_{self.restart_id}.restart"
        if not restart_file.exists():
            print(f"restart file {restart_file} does not exist.")
            exit()

        cmd = f"cp {restart_file} {self.wd / f'creep_init.restart'}"
        subprocess.call(cmd.split())

if __name__ == '__main__':
    d = RunCreep(2200, 100, 1)
    d.build()
