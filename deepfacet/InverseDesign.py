import sys, os, subprocess
sys.path.append("models/")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Paths import Paths
from DataPreprocessor import *
from CustomDataset import *
from LogAggregator import *
from cnn import Conv3D
import utils
from ConfigurationScreening import *
paths = Paths()



class InverseDesign:
    def __init__(self, generation_num:int, search_criteria:str, simulations_root_dir:str=None, target:float=None, num_pores:int=9, include_symmetries:bool=False):
        """
        pipeline for training, screening and testing a generation of the
        accelerated search algo ML model

        * generation_num:           generation number
        * search_criteria:          name of characteristic to search for (strongest/weakest/other)
        * simulations_root_dir:     path to root folder of randomly selected simulations
        * target:                   target yield stress (if search criteria is 'other')
        """
        self.root_dir = os.getcwd()
        self.generation_num = generation_num
        self.num_pores = num_pores
        if include_symmetries:
            search_criteria += "_with_syms"
        self.screened_sims_path = os.path.join(paths.COMPOSITE, f"DESIGN_{num_pores}_PORE", search_criteria)

        self.search_criteria = search_criteria
        self.include_symmetries = include_symmetries
        if not os.path.exists(self.screened_sims_path):
            print(f"{self.screened_sims_path} does not exist")


        if simulations_root_dir is None:
            self.simulations_root_dir = os.path.join(paths.COMPOSITE, f"DESIGN_{num_pores}_PORE")

        self.base_simulations_path = os.path.join(self.simulations_root_dir, "random_gen/completed_batches")

        self.model_gen_root_dir = os.path.join(paths.NETS, f"model_gens_{num_pores}_pore", search_criteria, f"0{generation_num}_gen")

        if not os.path.exists(self.model_gen_root_dir):
            print(f"new generation: {self.model_gen_root_dir}")
            os.makedirs(self.model_gen_root_dir)
        else:
            inp = input(f"this generation already exists. continue? YES/no:")
            if inp != "YES":
                print("exiting..")
                exit(1)


        self.device = utils.get_device("gpu")
        self.num_periodic = 2
        self.model_path = os.path.join(self.model_gen_root_dir, f"0{self.generation_num}_model.pth")
        self.is_trained = False



        # if (self.search_criteria != "strongest" and self.search_criteria != "weakest")\
        #     or (self.search_criteria != "strongest_with_syms" and self.search_criteria != "weakest_with_syms"):
        if (not "strongest" in self.search_criteria) and (not "weakest" in self.search_criteria):
            assert(target is not None), "arg 'target' cannot be 'None'"
            assert(isinstance(target, float))
        self.target = target

    def _init_model(self):
        #initiate CNN

        p = self.num_periodic*2 + 3
        self.model_params = {
            "input_shape": (1, p, p, p),
            "n_kernels": (32, 64, 128),
            "kernel_sizes": (4, 4, 4),
            "n_dense": 512,
            "padding": 1,
            "init": None,
        }

        self.model = Conv3D(**self.model_params, verbose=True).to(self.device)
        return self.model

    def _load_self(self):
        pass

    def _init_data(self, base_root_dir, base_batches):

        #add randomly generated config simulation results to batches
        batches = [os.path.join(base_root_dir, batch) for batch in base_batches]
        rerun_t0_base = [1700]*len(batches)
        rerun_t0_new = []

        # add previous generation screened sims to batches
        for gen in range(self.generation_num):
            batches.append(os.path.join(self.search_criteria, f"0{gen}_gen"))
            rerun_t0_new.append(1600)

        #aggregate simulation logs
        rerun_t0_mixed = rerun_t0_base + rerun_t0_new
        aggregator = LogAggregator(raw_data_path=self.simulations_root_dir, output_path=self.model_gen_root_dir)
        self.full_logs_fname = "full_logs.npz"
        if not os.path.exists(os.path.join(self.model_gen_root_dir, self.full_logs_fname)):
            aggregator.aggregate(self.full_logs_fname, batches=batches, rerun_t0=rerun_t0_mixed)

        #aggregate base simulation logs
        self.base_logs_fname = "base_logs.npz"
        batches = [os.path.join(base_root_dir, batch) for batch in base_batches]
        if not os.path.exists(os.path.join(self.model_gen_root_dir, self.base_logs_fname)):
            aggregator.aggregate(self.base_logs_fname, batches=batches, rerun_t0=rerun_t0_base)

        #create data for model
        self.model_data_fname = "model_input.npy"
        if not os.path.exists(os.path.join(self.model_gen_root_dir, self.model_data_fname)):
            aggregator.write_model_ready(self.full_logs_fname, self.model_data_fname)


        preprocessor_params = {
            "data_path": os.path.join(self.model_gen_root_dir, self.model_data_fname),
            "split_ratios": (0.80, 0.001, 0.199),  #train, val, test
            "include_symmetries":1,             #0 = no, 1 = rotational, 2 = rot + periodic
            "mode": "torch",
            "batch_size": 32,
            "predictor": "yield",
            "remove_outliers": None
        }

        self.criterion = nn.MSELoss()
        preprocessor = DataPreprocessor(**preprocessor_params)
        self.train_loader, self.val_loader, self.test_loader = \
            preprocessor.torch_load(type="3d", type_params={'num_periodic':self.num_periodic})

        self.val_loader = None

    def init_model_and_data(self, base_root_dir="random_gen/completed_batches", base_batches = ["01_batch", "02_batch", "03_batch"]):
        """
        prepare simulation data and initiate model
        returns model instance
        """
        files = os.listdir(self.model_gen_root_dir)
        if len(files) != 0:
            inp = input(f"{self.model_gen_root_dir} not empty. Double check class instance!!! Really continue? YES!!/no")
            if inp != "YES!!":
                print("Exiting..")
                exit()

        self._init_data(base_root_dir, base_batches)
        return self._init_model()

    def train_model(self, lr=1e-5, epochs=300):

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_model_params = {
            "model": self.model,
            "device": self.device,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "epochs": epochs,
            "criterion": self.criterion,
            "optimizer": optimizer,
            "save_model_path": None,
        }

        self.history, _ = utils.train_model(**train_model_params)
        self.is_trained = True
        print("Done training. State not saved!")

    def save_model_state(self):
        assert(self.is_trained)


        if os.path.exists(self.model_path):
            inp = input(f"{self.model_path} exists. Overwrite? YES/no:")
            if inp != "YES":
                print("Exiting..")
                return

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved model state to {self.model_path}")

    def test_model(self, plot_history=True, plot_predictions=True, save_history=True, save_predictions=True):
        if plot_history:
            utils.plot_history(self.history, train_data_only=True)

        type_dict = {
            "train" : self.train_loader,
            # "val": self.val_loader,
            "test": self.test_loader
        }

        for title, loader in type_dict.items():
            true, pred = utils.test_model(self.model, self.device, self.criterion, loader, plot_predictions=plot_predictions, title=title)
            if save_predictions:
                fname = os.path.join(self.model_gen_root_dir, f"true_vs_pred_{title}.npy")
                np.save(fname, np.stack((true, pred)))



        if plot_predictions:
            plt.show()

        if save_history:
            for data_type, vals in self.history.items():
                history_fname = os.path.join(self.model_gen_root_dir, f"history_{data_type}.npy")
                if os.path.exists(history_fname):
                    inp = input(f"{history_fname} exists. Overwrite? YES/no")
                    if inp != "YES":
                        return
                    np.save(history_fname, vals)

    def load_model(self):
        assert(os.path.exists(self.model_path)), f"{self.model_path} does not exist"
        print(f"Loading model {self.model_path}")
        self.model = self._init_model()
        model_state = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state)
        return self.model

    def screen_search_criteria(self, top_k = 10):
        if not self.is_trained:
            self.load_model()

        remove_symmetric = not self.include_symmetries
        db_path = os.path.join(paths.COMPOSITE, "coord_files_threes_new")


        self.screener = ConfigurationScreening(self.model,
                                               self.device,
                                               db_path,
                                               remove_symmetric=remove_symmetric,
                                               max_num=None,
                                               num_pores = self.num_pores,
                                               search_criteria=self.search_criteria)
        X = self.screener.to_model_format()

        # if self.search_criteria != "strongest" and self.search_criteria != "weakest":

        if "strongest" in self.search_criteria:
            property_type = "strongest"
        elif "weakest" in self.search_criteria:
            property_type = "weakest"
        else:
            property_type = "design"

        self.screening_results = self.screener.get_top_contenders(X, type=property_type, target=self.target, k=top_k)

    def get_contender_pore_nums(self, visualize=True):
        self.screened_nums = self.screener.get_pore_numbers(self.screening_results)
        if visualize:
            self.screener.visualize(self.screening_results, num_figs=10)

        return self.screened_nums

    def compare_simulations_with_predictions(self):

        #aggregate criterion logs (for screening)
        aggregator = LogAggregator(raw_data_path=self.screened_sims_path, output_path=self.model_gen_root_dir)
        self.screened_logs_fname = "screened_logs.npz"
        if not os.path.exists(os.path.join(self.model_gen_root_dir, self.screened_logs_fname)):
            aggregator.aggregate(self.screened_logs_fname, batches=[f"0{self.generation_num}_gen"], rerun_t0=[1600])
        else:
            print(f"Warning: not overwriting {os.path.join(self.model_gen_root_dir, self.screened_logs_fname)}")

        predictions = self.screening_results["y"]
        ConfigurationScreening.histogram(os.path.join(self.model_gen_root_dir, self.base_logs_fname),
                                         os.path.join(self.model_gen_root_dir, self.screened_logs_fname),
                                         predictions,
                                         np.mean(predictions))

        #
    def generate_screened_sims(self):
        self.root_dir = os.getcwd()
        os.chdir(paths.COMPOSITE)
        from ScriptGenerator import get_base_config, generate_screened_run

        if "strongest" in self.search_criteria:
            time = 2.8
        elif "weakest" in self.search_criteria:
            time = 2.2
        else:
            time = 2.4

        sim_params = {"time": time, "strain_rate": 0.03, "run_template": "runThrees_high_sr.lmp", **get_base_config()}
        target_dir = f"DESIGN_{self.num_pores}_PORE/{self.search_criteria}/0{self.generation_num}_gen"

        generate_screened_run(self.screened_nums, target_dir, **sim_params, search_criteria=self.search_criteria)

        subprocess.call(f"cp templates/autoqueue.py {target_dir}".split())

        os.chdir(self.root_dir)

    def run_all(self):
        self.init_model_and_data()
        self.train_model()
        self.save_model_state()
        self.test_model()
        self.screen_search_criteria()
        self.get_contender_pore_nums()
        self.generate_screened_sims()
        # self.compare_simulations_with_predictions()

















        #
