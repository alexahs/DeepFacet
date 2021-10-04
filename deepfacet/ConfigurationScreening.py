import torch
import numpy as np
import sys, os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../"))
from ThreesSystem import Three
from DataPreprocessor import DataPreprocessor
from cnn import Conv3D
import utils

class ConfigurationScreening:

    def __init__(self,
                 model,
                 device,
                 db_path:str="../../composite_systems/coord_files_threes_new",
                 batch_size=64,
                 periodic_padding:int=2,
                 max_num:int=None,
                 num_pores:int=9,
                 remove_symmetric:bool = True,
                 search_criteria:str = None):
        """
        model: pretrained model
        db_path: path to dir containing all possible configs and similated configs
        """
        path_generated = os.path.join(db_path, f"generated_systems_{num_pores}.txt")
        if remove_symmetric:
            path_full_space = os.path.join(db_path, f"pore_configs_{num_pores}.txt")
        else:
            path_full_space = os.path.join(db_path, f"all_pore_configs_{num_pores}.txt")


        if search_criteria is not None:
            path_search_criteria_generated = os.path.join(db_path, f"generated_systems_{num_pores}_{search_criteria}.txt")
        else:
            path_search_criteria_generated = None


        self.model = model
        self.model.to(device)
        self.device = device
        self.model.eval()
        self.max_num = max_num
        self.batch_size = batch_size
        self.num_pores = num_pores
        self.periodic_padding = periodic_padding
        self.remove_symmetric = remove_symmetric
        self.full_space = self.read_db(path_full_space, path_generated, path_search_criteria=path_search_criteria_generated)

    def read_db(self, path_full_space, path_simulated, path_search_criteria):
        """
        returns a set of all possible pore configuration numbers (less simulated ones)
        """
        #load base simulated configurations
        prev_generated = set()
        try:
            tmp0 = np.genfromtxt(path_simulated, delimiter=',', dtype=np.uint, skip_header=6)[:,1]
            for config in tmp0:
                prev_generated.add(config)
        except:
            print(f"{path_simulated} does not exist")

        if path_search_criteria is not None and os.path.exists(path_search_criteria):
            # try:
            # tmp1 = np.genfromtxt(path_search_criteria, delimiter=',', dtype=np.uint, skip_header=6)[:,1]
            tmp1 = np.genfromtxt(path_search_criteria, delimiter=',', dtype=np.uint, skip_header=6)
            if len(tmp1) != 0:
                for config in tmp1[:,1]:
                    prev_generated.add(config)
            # except:
                # print(f"{path_search_criteria} does not exist")

        #load full configuration space
        tmp2 = np.genfromtxt(path_full_space, delimiter=',', dtype=np.uint, skip_header=16)

        full_map = {}
        for i in range(tmp2.shape[0]):
            full_map[i+1] = tmp2[i]

        search_space = {}

        print("Loading configs..")
        for i in tqdm(full_map.keys()):
            if i not in prev_generated:
                search_space[i] = full_map[i]

        return search_space

    def to_model_format(self, cached_data_path="./", check_cached=False):

        n = len(self.full_space)
        # if not self.remove_symmetric:
        #     cached_fname = os.path.join(cached_data_path, f"screening_data_{self.num_pores}_pores_with_syms.npz")
        # else:
        #     cached_fname = os.path.join(cached_data_path, f"screening_data_{self.num_pores}_pores_no_syms.npz")
        # if os.path.exists(cached_fname) and check_cached:
        #     X = np.load(cached_fname, allow_pickle=True)['arr_0'][()]
        #     if X.shape[0] == n:
        #         if self.max_num is None:
        #             return X
        #         else:
        #             X = X[:self.max_num]
        #             np.random.shuffle(X)
        #             return X

        p = self.periodic_padding
        pp = 3+2*p
        X = np.zeros((n, 3, 3, 3), dtype=np.float32)
        print("Converting formats..")
        i = 0
        for config_nums in tqdm(self.full_space.values()):
            bool = Three.nums_to_n_hot(config_nums)
            X[i] = np.reshape(bool, (3, 3, 3))
            i += 1


        X = DataPreprocessor.pad_X(X, p)
        X = X.reshape(X.shape[0], 1, pp, pp, pp).astype(np.float32)

        # np.savez(cached_fname, X, allow_pickle=True)
        if self.max_num is None:
            return X
        else:
            X = X[:self.max_num]
            np.random.shuffle(X)
            return X

    def from_bool_to_model_fmt(self, X):

        p = self.periodic_padding
        pp = 3+2*p
        X = np.zeros((n, 3, 3, 3), dtype=np.float32)
        print("Converting formats..")
        for i, config_nums in enumerate(self.full_space.values()):
            bool = Three.nums_to_n_hot(config_nums)
            X[i] = np.reshape(bool, (3, 3, 3))


        X = DataPreprocessor.pad_X(X, p)
        X = X.reshape(X.shape[0], 1, pp, pp, pp).astype(np.float32)

    def _predict(self, X):
        n = X.shape[0]
        n_batches_remain = n % self.batch_size
        n_batches = n // self.batch_size
        predictions = np.zeros(n)

        end = None
        batches = trange(n_batches)
        print("Searching..")
        with torch.no_grad():
            for i in batches:

                beg = i * self.batch_size
                end = beg + self.batch_size

                to_predict = torch.from_numpy(X[beg:end]).to(self.device)
                pred = self.model(to_predict).cpu().numpy()
                predictions[beg:end] = pred

                # if i % n//100 == 0:
                #     max = np.max(predictions[:end])
                #     min = np.min(predictions[:end])
                #     batches.set_postfix({"Strongest": f"{max:.2f}", "Weakest": f"{min:.2f}"})

            if n % self.batch_size != 0:
                if end is None:
                    # beg = 0
                    end = 0
                to_predict = torch.from_numpy(X[end:]).to(self.device)
                pred = self.model(to_predict)
                predictions[end:] = pred.cpu().numpy()
                # max = np.max(predictions[:end])
                # min = np.min(predictions[:end])
                # batches.set_postfix({"Strongest": f"{max:.2f}", "Weakest": f"{min:.2f}"})


        return predictions

    def get_designed(self, X, target, k=20):
        """
        X:      input model data
        target: target stress values
        k:      number of configs to return
        """

        predictions = self._predict(X)
        distance = np.abs(predictions - target)
        inds = np.argpartition(distance, k)[:k]

        X_target = np.squeeze(X[inds], axis=1)

        results = {
            "X": X_target,
            "y": predictions[inds]
        }

        return results

    def get_top_contenders(self, X, target, type, k=20):
        predictions = self._predict(X)

        if type == "design":
            distance = np.abs(predictions - target)
            inds = np.argpartition(distance, k)[:k]
            X_target = np.squeeze(X[inds], axis=1)
            y_target = predictions[inds]

        if type == "strongest":
            inds = np.argpartition(predictions, -k)[-k:]
            inds_sorted = np.flip(inds[np.argsort(predictions[inds])])
            X_target = np.squeeze(X[inds_sorted], axis=1)
            y_target = predictions[inds_sorted]

        if type == "weakest":
            inds = np.argpartition(predictions, k)[:k]
            inds_sorted = inds[np.argsort(predictions[inds])]
            X_target = np.squeeze(X[inds_sorted], axis=1)
            y_target = predictions[inds_sorted]

        results = {
            "X": X_target,
            "y": y_target
        }

        return results

    def get_extremes(self, X, k):
        """
        X: model data
        k_top: k strongest/weakest to return
        """
        if self.max_num is not None:
            if k_top >= self.max_num:
                k_top = self.max_num-1

        predictions = self._predict(X)

        # get top k
        inds_best = np.argpartition(predictions, -k)[-k:]
        inds_best_sorted = np.flip(inds_best[np.argsort(predictions[inds_best])])

        inds_worst = np.argpartition(predictions, k)[:k]
        inds_worst_sorted = inds_worst[np.argsort(predictions[inds_worst])]

        strongest_y = predictions[inds_best_sorted]
        weakest_y = predictions[inds_worst_sorted]

        strongest_X = np.squeeze(X[inds_best_sorted], axis=1)
        weakest_X = np.squeeze(X[inds_worst_sorted], axis=1)

        results = {
            "strongest_X": strongest_X,
            "strongest_y": strongest_y,
            "weakest_X": weakest_X,
            "weakest_y": weakest_y
        }
        return results

    def visualize(self, results, num_figs = 1, num_points=11, figsize=None):
        from UpscaleData import BoolToImage

        """
        results: result dict from self.predict()
        n : number of points per grid cell
        """
        X = results["X"]
        Y = results["y"]
        n = num_points
        if num_figs > X.shape[0]:
            num_figs = X.shape[0]

        painter = BoolToImage(n, 0.4)

        X = DataPreprocessor.periodic_3d_to_bool(X, self.periodic_padding)

        nx, ny, nz = 3*n, 3*n, 3

        images = np.zeros((num_figs, nx, ny, nz))
        for i in range(num_figs):
            for dim in range(3):
                dim_bool = X[i,dim*9:(dim+1)*9]
                grid = np.rot90(np.reshape(dim_bool, (3, 3)), k=0)
                for x in range(3):
                    for y in range(3):
                        cell = grid[x, y]
                        if cell == 1:
                            img = painter.draw()
                        else:
                            img = np.zeros((n, n))
                        images[i, x*n:(x+1)*n, y*n:(y+1)*n, dim] = img


        if figsize is None:
            figsize = (5, num_figs*2)

        fig, ax = plt.subplots(nrows = num_figs, ncols = 3, figsize=figsize)
        for i in range(num_figs):
            for j in range(3):
                ax[i, j].imshow(images[i, :, :, j])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            ax[i, 1].set_title(f"Yield Strength: {Y[i]:.2f}")
        plt.show()

    def get_pore_numbers(self, results):
        sys.path.append("../")
        from Symmetries import Symmetries
        pore_nums = []
        key = "X"

        for X in results[key]:
            onehot = DataPreprocessor.periodic_3d_to_bool(X, self.periodic_padding)
            nums = Symmetries.n_hot_to_nums(onehot)
            pore_nums.append(nums)

        s = "["
        for num in pore_nums:
            s += f"{num},\n"
        s = s[:-2]
        s += "]"
        print(f"Top contender pore numbers:")
        print(s)

        return pore_nums

    @staticmethod
    def histogram(fname_full_data, fname_screened_data, predictions, target_value):
        from scipy.stats import norm
        yield_full = []
        yield_screened = []

        data_full = np.load(fname_full_data, allow_pickle=True)['arr_0'][()]
        data_screened = np.load(fname_screened_data, allow_pickle=True)['arr_0'][()]

        for d in data_full.values():
            yield_full.append(d['mean_yield'])

        for d in data_screened.values():
            yield_screened.append(d['mean_yield'])


        mu_sim = np.mean(yield_screened)
        std_sim = np.std(yield_screened)
        mu_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        print(f"Simulated: {mu_sim:.3f} +- {std_sim:.3f}")
        print(f"Target: {mu_pred:.3f} +- {std_pred:.3f}")

        print("max:", np.max(yield_screened))
        print("min:", np.min(yield_screened))

        if std_pred < 0.001:
            predictions = np.random.normal(mu_pred, std_pred+0.01, predictions.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["tab:blue", "tab:red", "tab:green"]
        n_bins = [25, 20, 3]
        labels = ["Randomly selected",
                  r"Simulated, $\overline{\sigma} = $" + f"{mu_sim:.3f}" + r"$\pm$" + f"{std_sim:.3f}",
                  r"Target, $\overline{\sigma}$ = " + f"{mu_pred:.3f}" + r"$\pm$" + f"{std_pred:.3f}"]


        for i, stress in enumerate([yield_full, yield_screened, predictions]):
            counts, bins = np.histogram(stress, bins=n_bins[i])
            ax.hist(bins[:-1], bins, density=False, weights=counts, color=colors[i], alpha=0.5, label=labels[i])



        xvals = np.linspace(mu_sim - 3*std_sim, mu_sim + 3*std_sim)
        ax.plot(xvals, norm.pdf(xvals, mu_sim, std_sim), color=colors[1])
        ax.set_xlabel("Yield Stress [GPa]")
        ax.legend()
        ax.set_ylabel("Counts")
        plt.show()
