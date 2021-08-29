from sklearn.model_selection import KFold
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import utils
from dnn import Dnn, He_init_DNN
from cnn import Conv3D, He_init_CNN
from DataPreprocessor import *
from CustomDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import itertools as it
from tqdm import trange
from dataclasses import dataclass
from typing import OrderedDict
from CVResults import *


class CrossValidation:
    """
    Base class for performing cross validation of a pytorch model for a set of architectural parameters
    Params:
    * search_params: dict[param_name] = [params]
    * model_cls: reference to pytorch model class
    * train_func: function for training the model
    * devide: pytorch device
    * mode: scoring criteria, 'mse' or 'r2' score
    """
    def __init__(self, search_params, model_cls, train_func, device, mode = "mse"):

        self.search_params = search_params
        self.model_cls = model_cls
        self.train_model = train_func
        self.device = device
        assert(mode in ["r2", "mse"])
        self.mode = mode
        self.criterion = nn.MSELoss()

        self.num_runs = 1
        for val in self.search_params.values():
            self.num_runs *= len(val)

    def select_params(self, i):
        raise NotImplementedError

    def _fit(self, params, X, y, epochs, kfold_splits):
        raise NotImplementedError

    def fit(self, X, y, epochs=300, kfold_splits=5, verbose=False):
        """
        X, y: train, val
        """

        if verbose:
            pbar = trange(self.num_runs)
        else:
            pbar = range(self.num_runs)

        best_r2 = np.NINF
        best_mse = np.Inf
        best_inds = None
        best_fit_result = None
        best_params = None
        results = []

        for i in pbar:
            params, param_indices = self.select_params(i)
            fit_results = self._fit(params, X, y, epochs, kfold_splits)

            r2 = fit_results["r2_val"]
            mse = fit_results["mse_val"]
            if self.mode == "r2":
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit_result = fit_results
                    best_inds = param_indices
                    best_params = params

            elif self.mode == "mse":
                if mse < best_mse:
                    best_mse = mse
                    best_fit_result = fit_results
                    best_inds = param_indices
                    best_params = params

            result = Result(
                r2_train = fit_results["r2_train"],
                r2_val = fit_results["r2_val"],
                mse_train = fit_results["mse_train"],
                mse_val = fit_results["mse_val"],
                model_params = params
            )

            results.append(result)

            pbar_items = {
                "current_r2": r2,
                "best_r2": best_r2
            }

            if verbose:
                pbar.set_postfix(pbar_items)

        return best_inds, best_fit_result, best_params, reversed(sorted(results))


class GridSearch(CrossValidation):
    """
    performs a sweeping grid search over a set of architechtural parameters
    """
    def __init__(self, search_params, model_cls, train_func, device, mode):
        super().__init__(search_params, model_cls, train_func, device, mode)
        self.build_param_combinations()
        self.print_info()

    def build_param_combinations(self):
        self.param_combinations = []
        self.param_indices = []
        ranges = []
        keys = []
        for k, v in self.search_params.items():
            ranges.append(range(len(v)))
            keys.append(k)

        for indices in it.product(*ranges):
            tmp_dict = {}
            self.param_indices.append(indices)
            for i, param_idx in enumerate(indices):
                tmp_dict[keys[i]] = self.search_params[keys[i]][param_idx]
            self.param_combinations.append(tmp_dict)

    def select_params(self, i):
        return self.param_combinations[i], self.param_indices[i]

    def print_info(self):
        print(f"{self.__class__.__name__}: total params: {self.num_runs}")


class GridSearchCNN(GridSearch):
    def __init__(self, search_params, model_cls, train_func, device, mode):
        super().__init__(search_params, model_cls, train_func, device, mode)

    def _fit(self, params, X, y, epochs, kfold_splits):

        batch_size = params.get("batch_size") or 32
        learning_rate = params.get("learning_rate") or 1e-5


        kernel_sizes = []
        for i in range(len(params["n_kernels"])):
            kernel_sizes.append(params["kernel_sizes"])
        kernel_sizes = tuple(kernel_sizes)

        model_params = {
            "input_shape": X[0,:].shape,
            "n_kernels": params["n_kernels"],
            "kernel_sizes": kernel_sizes,
            "n_dense": params["n_dense"],
            "padding": 1,
            "init": None,
            "bias": params["bias"]
        }
        count = 0
        r2_scores = []

        best_train_loader = None
        best_val_loader = None
        r2_local_best = np.NINF
        mse_local_best = np.Inf

        best_model_state = None
        best_history = None


        metrics = {
            "r2_train": [],
            "r2_val": [],
            "mse_train": [],
            "mse_val": []
        }


        for train_idx, val_idx in KFold(n_splits=kfold_splits, random_state=0, shuffle=True).split(X):

            data_train = CustomDataset(X[train_idx], y[train_idx])
            data_val = CustomDataset(X[val_idx], y[val_idx])

            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

            model = self.model_cls(**model_params, verbose=False).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_model_params = {
                "model": model,
                "device": self.device,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "epochs": epochs,
                "criterion": self.criterion,
                "optimizer": optimizer,
                "save_model_path": None
            }


            history, model_state = self.train_model(**train_model_params)

            true_train, pred_train = utils.test_model(model, self.device, self.criterion, train_loader, plot_predictions=False, verbose=False, title="train")
            true_val, pred_val = utils.test_model(model, self.device, self.criterion, val_loader, plot_predictions=False, verbose=False, title="val")

            r2_train = utils.r2_score(pred_train, true_train)
            r2_val = utils.r2_score(pred_val, true_val)

            mse_train = utils.MSE(pred_train, true_train)
            mse_val = utils.MSE(pred_val, true_val)

            metrics["r2_train"].append(r2_train)
            metrics["r2_val"].append(r2_val)
            metrics["mse_train"].append(mse_train)
            metrics["mse_val"].append(mse_val)

            if self.mode == "r2":
                if r2_val > r2_local_best:
                    best_model_state = model_state
                    best_history = history
                    r2_local_best = r2_val
                    mse_local_best = mse_val
            elif self.mode == "mse":
                if mse_val < mse_local_best:
                    best_model_state = model_state
                    best_history = history
                    mse_local_best = mse_val
                    r2_local_best = r2_val

        mean_r2 = np.mean(metrics["r2_val"])
        mean_mse = np.mean(metrics["mse_val"])

        results = {
            "model": model,
            "model_state": best_model_state,
            "r2_val": np.mean(metrics["r2_val"]),
            "mse_val": np.mean(metrics["mse_val"]),
            "r2_train": np.mean(metrics["r2_train"]),
            "mse_train": np.mean(metrics["mse_train"]),
            "history": best_history
        }

        return results

class GridSearchDNN(GridSearch):
    def __init__(self, search_params, model_cls, train_func, device, mode):
        super().__init__(search_params, model_cls, train_func, device, mode)

    def _fit(self, params, X, y, epochs, kfold_splits):

        batch_size = params.get("batch_size") or 32
        learning_rate = params.get("learning_rate") or 1e-5

        model_params = {
            "input_shape": X[0].shape[0],
            "n_layers": params["n_layers"],
            "n_nodes": params["n_nodes"],
            "bias": params["bias"]
        }

        count = 0
        r2_scores = []

        best_train_loader = None
        best_val_loader = None
        r2_local_best = np.NINF
        mse_local_best = np.Inf

        best_model_state = None
        best_history = None


        metrics = {
            "r2_train": [],
            "r2_val": [],
            "mse_train": [],
            "mse_val": []
        }


        for train_idx, val_idx in KFold(n_splits=kfold_splits, random_state=0, shuffle=True).split(X):

            data_train = CustomDataset(X[train_idx], y[train_idx])
            data_val = CustomDataset(X[val_idx], y[val_idx])

            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

            model = self.model_cls(**model_params, verbose=False).to(self.device)
            model.apply(He_init_DNN)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_model_params = {
                "model": model,
                "device": self.device,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "epochs": epochs,
                "criterion": self.criterion,
                "optimizer": optimizer,
                "save_model_path": None
            }

            history, model_state = self.train_model(**train_model_params)

            true_train, pred_train = utils.test_model(model, self.device, self.criterion, train_loader, plot_predictions=False, verbose=False, title="train")
            true_val, pred_val = utils.test_model(model, self.device, self.criterion, val_loader, plot_predictions=False, verbose=False, title="val")

            r2_train = utils.r2_score(pred_train, true_train)
            r2_val = utils.r2_score(pred_val, true_val)

            mse_train = utils.MSE(pred_train, true_train)
            mse_val = utils.MSE(pred_val, true_val)

            metrics["r2_train"].append(r2_train)
            metrics["r2_val"].append(r2_val)
            metrics["mse_train"].append(mse_train)
            metrics["mse_val"].append(mse_val)

            if self.mode == "r2":
                if r2_val > r2_local_best:
                    best_model_state = model_state
                    best_history = history
                    r2_local_best = r2_val
                    mse_local_best = mse_val
            elif self.mode == "mse":
                if mse_val < mse_local_best:
                    best_model_state = model_state
                    best_history = history
                    mse_local_best = mse_val
                    r2_local_best = r2_val

        mean_r2 = np.mean(metrics["r2_val"])
        mean_mse = np.mean(metrics["mse_val"])

        results = {
            "model": model,
            "model_state": best_model_state,
            "r2_val": np.mean(metrics["r2_val"]),
            "mse_val": np.mean(metrics["mse_val"]),
            "r2_train": np.mean(metrics["r2_train"]),
            "mse_train": np.mean(metrics["mse_train"]),
            "history": best_history
        }

        return results

class CVDataDNN:
    def __init__(self, data_preprocessor, num_periodic=None):
        dp = data_preprocessor
        X = dp.X
        y = dp.y

        if dp.include_symmetries > 0:
            X, y = dp.get_symmetries(dp.X, dp.y)

        if num_periodic is not None:
            X = X.reshape(X.shape[0], 3, 3, 3)
            X = dp.pad_X(X, num_periodic)
            new_dim = (3+2*num_periodic)**3
            X = X.reshape(X.shape[0], new_dim)

        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)

        X = X[inds]
        y = y[inds]

        n_CV = int((dp.train_ratio + dp.val_ratio)*X.shape[0])

        self.X_CV = X[0:n_CV]
        self.y_CV = y[0:n_CV]

        self.X_test = X[n_CV:-1]
        self.y_test = y[n_CV:-1]

        print("========DATA LOADED========")
        print(f"Target: {dp.predictor} stress")
        print(f"Number of samples : {self.X_CV.shape[0]} CV (train + val), {self.X_test.shape[0]} test")
        print(f"Sample shape: {self.X_CV.shape[1:]}")
        print("===========================")

    def get(self):
        return self.X_CV, self.y_CV, self.X_test, self.y_test

class CVDataCNN(CVDataDNN):
    def __init__(self, dataPreprocessor, num_periodic=None):
        dp = dataPreprocessor

        X = dp.X
        y = dp.y

        if dp.include_symmetries > 0:
            X, y = dp.get_symmetries(X, y)

        X = X.reshape(X.shape[0], 3, 3, 3)

        if num_periodic is not None:
            X = dp.pad_X(X, num_periodic)
            new_dim = 3+2*num_periodic
        else:
            new_dim = 3

        if dp.mode == "tf":
            X = X.reshape(X.shape + (1,))
        elif dp.mode == "torch":
            X = X.reshape(X.shape[0], 1, new_dim, new_dim, new_dim)

        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)

        X = X[inds]
        y = y[inds]

        n_CV = int((dp.train_ratio + dp.val_ratio)*X.shape[0])

        self.X_CV = X[0:n_CV]
        self.y_CV = y[0:n_CV]

        self.X_test = X[n_CV:-1]
        self.y_test = y[n_CV:-1]

        print("========DATA LOADED========")
        print(f"Target: {dp.predictor} stress")
        print(f"Number of samples : {self.X_CV.shape[0]} CV (train + val), {self.X_test.shape[0]} test")
        print(f"Sample shape: {self.X_CV.shape[1:]}")
        print("===========================")

def run_dnn_search(epochs, mode):

    outname = f"CV_results/scores_dnn.npz"
    if os.path.exists(outname):
        print(f"WARNING: {outname} exists. Exiting..")
        return
    else:
        print(f"running search, saving to {outname}")
    preprocessor_params = {
        "data_path": "../data/model_input.npy",
        "split_ratios": (0.6, 0.2, 0.2),  #train, val, test
        "include_symmetries":1,             #0 = no, 1 = rotational, 2 = rot + periodic
        "mode": "torch",
        "batch_size": None,
        "predictor": "yield",
        "remove_outliers": None
    }
    preprocessor = DataPreprocessor(**preprocessor_params)
    num_periodic = 2
    X_CV, y_CV, X_test, y_test = CVDataDNN(preprocessor, num_periodic = num_periodic).get() #X_CV, y_CV, X_test, y_test
    device = utils.get_device("cpu")

    n_nodes_list = 2**np.arange(2, 11) # 4 - 1024 nodes
    n_layers_list = 2**np.arange(1, 8) # 2 - 128 layers

    search_params = {
        "n_nodes": n_nodes_list,
        "n_layers": n_layers_list,
        "learning_rate": [1e-5],
        "batch_size": [32]
    }

    splits = 5

    gridsearch = GridSearchDNN(search_params, Dnn, utils.train_model, device, mode = mode)
    best_inds, best_instance_vars, final_params, results = gridsearch.fit(X_CV, y_CV, epochs, splits, verbose=True)

    model = best_instance_vars["model"]
    test_loader = DataLoader(CustomDataset(X_test, y_test))
    history = best_instance_vars["history"]
    test_true, test_pred = utils.test_model(model, device, nn.MSELoss, test_loader, title="test")
    r2_test = utils.r2_score(test_pred, test_true)
    mse_test = utils.MSE(test_pred, test_true)

    final_state = best_instance_vars["model_state"]
    print(f"{r2_test=}")
    print(f"{mse_test=}")
    print(f"{best_inds=}")
    print(f"{final_params=}")

    final_result = FinalResult(r2_test, mse_test, final_params, final_state, test_true, test_pred, history)
    results = list(results)
    results.insert(0, final_result)

    np.savez(outname, results, allow_pickle=True)
    print(f"wrote {outname}")

def run_cnn_search(epochs, mode):

    outname = f"CV_results/scores_cnn.npz"
    if os.path.exists(outname):
        print(f"WARNING: {outname} exists. Exiting..")
        return
    else:
        print(f"running search, saving to {outname}")
    preprocessor_params = {
        "data_path": "../data/model_input.npy",
        "split_ratios": (0.6, 0.2, 0.2),  #train, val, test
        "include_symmetries":1,             #0 = no, 1 = rotational, 2 = rot + periodic
        "mode": "torch",
        "batch_size": None,
        "predictor": "yield",
        "remove_outliers": None
    }
    preprocessor = DataPreprocessor(**preprocessor_params)
    num_periodic = 2
    X_CV, y_CV, X_test, y_test = CVDataCNN(preprocessor, num_periodic = num_periodic).get() #X_CV, y_CV, X_test, y_test
    device = utils.get_device("gpu")

    kernel_size_list = [3, 4, 5]
    n_kernels_list = [(8, 16, 32), (16, 32, 64), (32, 64, 128)]
    n_dense_list = 2**np.arange(2, 11)

    search_params = {
        "kernel_sizes": kernel_size_list,
        "n_kernels": n_kernels_list,
        "n_dense": n_dense_list,
        "learning_rate": [1e-5],
        "batch_size": [32]
    }

    splits = 5

    gridsearch = GridSearchCNN(search_params, Conv3D, utils.train_model, device, mode = mode)
    best_inds, best_instance_vars, final_params, results = gridsearch.fit(X_CV, y_CV, epochs, splits, verbose=True)

    model = best_instance_vars["model"]
    test_loader = DataLoader(CustomDataset(X_test, y_test))
    history = best_instance_vars["history"]
    test_true, test_pred = utils.test_model(model, device, nn.MSELoss, test_loader, title="test")
    r2_test = utils.r2_score(test_pred, test_true)
    mse_test = utils.MSE(test_pred, test_true)
    final_state = best_instance_vars["model_state"]
    print(f"{r2_test=}")
    print(f"{mse_test=}")
    print(f"{best_inds=}")
    print(f"{final_params=}")

    final_result = FinalResult(r2_test, mse_test, final_params, final_state, test_true, test_pred, history)
    results = list(results)
    results.insert(0, final_result)
    np.savez(outname, results, allow_pickle=True)
    print(f"wrote {outname}")

def main():

    if len(sys.argv) < 2:
        print("no arg, exiting")
        exit()

    epochs = 300
    mode = "mse"
    nn_type = sys.argv[1]
    if nn_type == "cnn":
        run_cnn_search(epochs=epochs, mode=mode)
    elif nn_type == "dnn":
        run_dnn_search(epochs=epochs, mode=mode)
    else:
        print(f"{nn_type=} not valid. specify 'cnn' or 'dnn'")

if __name__ == '__main__':
    main()
