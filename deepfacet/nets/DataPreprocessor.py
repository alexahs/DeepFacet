import numpy as np
import sys, os
import math

class DataPreprocessor:
    def __init__(self, data_path, split_ratios:tuple, include_symmetries:int=1, remove_outliers:int=None, batch_size:int=None, predictor:str="yield", mode:str="torch"):
        """
        Loads a datafile of shape [n, 27+2] where the 27 first entries are the boolean pore placements
        and the last two entries are the yield and residual stresses respectively.
        n is the number of samples.

        *split_ratio: ratios of (train, val, test)
        *include_symmetries:
            0: no,
            1: rotational,
            2: rotational and periodic

        """
        if predictor != "yield" and predictor != "residual":
            print("Invalid predictor argument. Accepted values are \"yield\" and \"residual\"")
            exit(1)

        if include_symmetries not in [0, 1, 2]:
            print("Invalid arg for \"include_symmetries\". Valid options are 1, 2 or 3")
            exit(1)

        if mode != "torch" and mode != "tf":
            print("Error: mode must be set as \"tf\" or \"torch\"")
            exit(1)


        if split_ratios is not None:
            assert(np.sum(split_ratios) > 0.999), \
                f"Split ratios must sum to 1.. {self.train_ratio}+{self.val_ratio}+{self.test_ratio} != 1"
            self.train_ratio, self.val_ratio, self.test_ratio = split_ratios

        self.split_ratios = split_ratios
        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = mode
        self.predictor = predictor
        self.include_symmetries = include_symmetries
        self.include_periodic = True if include_symmetries == 2 else False
        self.data = np.load(self.data_path, allow_pickle=True)
        self.input_data_shape = None
        input_idx = -2
        if predictor == "yield":
            target_idx = -1
        elif predictor == "residual":
            target_idx = -2

        self.X = self.data[:,:input_idx]
        self.y = self.data[:,target_idx]

        if remove_outliers is not None:
            inds = DataPreprocessor.remove_outliers(self.y, remove_outliers)
            self.X = self.X[inds]
            self.y = self.y[inds]

    @staticmethod
    def pad_X(X, p):
        """
        p = pad width
        """
        if len(X.shape) > 3:
            X_new = np.zeros((X.shape[0], 3+2*p, 3+2*p, 3+2*p))
            for i in range(X.shape[0]):
                X_new[i] = np.pad(X[i], [(p,p), (p,p), (p,p)], 'wrap')
            return X_new

        else:
            return np.pad(X, [(p,p), (p,p), (p,p)], 'wrap')

    @staticmethod
    def periodic_3d_to_bool(X, p):
        """
        p = pad width
        """
        if len(X.shape) > 3:
            X_new = np.zeros((X.shape[0], 27))
            for i in range(X.shape[0]):
                X_new[i] = np.ravel(X[i, p:-p, p:-p, p:-p])
            return X_new.astype(np.int)
        else:
            return np.ravel(X[p:-p, p:-p, p:-p]).astype(np.int)

    @staticmethod
    def remove_outliers(x, std_limit=2.5):
        """
        returns indices of x which are within std_limit standard deviations

        x: target stress
        std_limit: keep data within this many std
        """

        mean = np.mean(x)
        std = np.std(x)
        x_sub_mean = np.abs(x-mean)
        keep_idx = np.where(x_sub_mean < std_limit*std)[0]

        print("======OUTLIER CLEANSING======")
        print(f"Mean value: {mean:.2f}")
        print(f"Standard deviation: {std:.2f}")
        print(f"Removal limit: {std_limit*std:.2f} ({std_limit:.1f}*std)")
        print(f"Number of removed instances: {len(x) - len(keep_idx)}")
        print(f"Number of retained instances: {len(keep_idx)}")
        print("=============================")

        return keep_idx

    def split_samples(self, X, y):
        if self.split_ratios is None:
            self.input_data_shape = X[0,:].shape
            return X, y
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - self.train_ratio, shuffle=True)
            X_val, X_test, y_val, y_test = \
                train_test_split(X_test, y_test,test_size=self.test_ratio/(self.test_ratio + self.val_ratio), shuffle=True)


            if self.mode == "tf" and self.batch_size is not None:
                b1 = X_train.shape[0] - X_train.shape[0] % self.batch_size
                b2 = X_val.shape[0] - X_val.shape[0] % self.batch_size
                b3 = X_test.shape[0] - X_test.shape[0] % self.batch_size
            else:
                b1 = X_train.shape[0]
                b2 = X_val.shape[0]
                b3 = X_test.shape[0]

            print("========DATA LOADED========")
            print(f"Target: {self.predictor} stress")
            print(f"Number of samples : {X_train[:b1].shape[0]} train, {X_val[:b2].shape[0]} validation, {X_test[:b3].shape[0]} test")
            print(f"Sample shape: {X_train.shape[1:]}")
            print("===========================")


            self.input_data_shape = X_train[0,:].shape

            return X_train[:b1], X_val[:b2], X_test[:b3], y_train[:b1], y_val[:b2], y_test[:b3]

    def load_3D_data(self, num_periodic:int=None):
        """
        Loads a datafile of shape [n, 27+2] where the 27 first entries are the boolean pore placements
        and the last two entries are the yield and residual stresses respectively.
        n is the number of samples.

        returns:
        * X_train, X_val, X_test:
            - model input data: arrays of size [N, 3+2p, 3+2p, 3+2p, 1] where p is the periodic padding number
        * y_train, y_val, y_test:
            - model target data: arrays of size [N, 1]

        (N = n_train, n_val etc.)
        """
        if self.include_symmetries > 0:
            # print(self.X, self.y, self.include_periodic)
            print("include_periodic:", self.include_periodic)
            # print(type(self.X), type(self.y), type(self.include_periodic))
            X, y = DataPreprocessor.get_symmetries(self.X, self.y, include_periodic=self.include_periodic, include_orig=True)
        else:
            X, y = self.X, self.y

        X = X.reshape(X.shape[0], 3, 3, 3)

        if num_periodic is not None:
            X = self.pad_X(X, num_periodic)
            new_dim = 3+2*num_periodic
        else:
            new_dim = 3

        if self.mode == "tf":
            X = X.reshape(X.shape + (1,))
        elif self.mode == "torch":
            X = X.reshape(X.shape[0], 1, new_dim, new_dim, new_dim)

        return self.split_samples(X, y)

    def load_1D_data(self, num_periodic:int=None):
        """
        Loads a datafile of shape [n, 27+2] where the 27 first entries are the boolean pore placements
        and the last two entries are the yield and residual stresses respectively.
        n is the number of samples.

        returns:
        * X_train, X_val, X_test:
            - model input data: arrays of size [N, 27]
        * y_train, y_val, y_test:
            - model target data: arrays of size [N, 1]

        (N = n_train, n_val etc.)
        """
        X, y = self.X, self.y

        if self.include_symmetries > 0:
            X, y = DataPreprocessor.get_symmetries(self.X, self.y, include_periodic=self.include_periodic)
        else:
            X, y = self.X, self.y

        if num_periodic is not None:
            X = X.reshape(X.shape[0], 3, 3, 3)
            X = self.pad_X(X, num_periodic)
            new_dim = (3+2*num_periodic)**3
            X = X.reshape(X.shape[0], new_dim)

        return self.split_samples(X, y)

    @staticmethod
    def get_symmetries(X, y, include_periodic=False, include_orig=True):
        sys.path.append("../")
        from Symmetries import Symmetries
        sym_dict = Symmetries.generate_rotational_symmetric(X, periodic=include_periodic)
        new_X = []
        new_y = []
        for i in range(X.shape[0]):
            repr = tuple(X[i,:])
            target = y[i]
            for syms in sym_dict[repr]:
                new_X.append(list(syms))
                new_y.append(target)
            if include_orig:
                new_X.append(np.array(X[i,:], dtype=np.uint8))
                new_y.append(target)


        X = np.array(new_X)
        y = np.array(new_y)
        return X, y

    def torch_load(self, type:str, type_params:dict):
        from CustomDataset import CustomDataset
        from torch.utils.data import DataLoader

        if type == "1d":
            data = self.load_1D_data(**type_params)
        elif type == "3d":
            data = self.load_3D_data(**type_params)
        else:
            print("Invalid argument for \"type\". Valid options are: \"1d\" or \"3d\"")
            exit(1)


        if self.batch_size is None:
            batch_size = 1
        else:
            batch_size = self.batch_size

        if self.split_ratios is None:
            X, y = data
            data_full = CustomDataset(X, y)
            loader = DataLoader(data_full, batch_size=batch_size, shuffle=False)
            return loader
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = data

            data_train = CustomDataset(X_train, y_train)
            data_val = CustomDataset(X_val, y_val)
            data_test = CustomDataset(X_test, y_test)

            #shuffling is done in sklearn.train_test_split
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader, test_loader

    def get_input_shape(self):
        return self.input_data_shape

class CVData:
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


    def get_data(self):
        return self.X_CV, self.y_CV, self.X_test, self.y_test

class CVDataDNN(CVData):
    def __init__(self, dataPreprocessor, num_periodic=None):
        dp = dataPreprocessor
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











#
