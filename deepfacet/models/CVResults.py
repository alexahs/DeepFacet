class Result:
    def __init__(self,
                r2_train:float,
                r2_val:float,
                mse_train:float,
                mse_val:float,
                model_params:dict):

        self.r2_train = r2_train
        self.r2_val = r2_val
        self.mse_train = mse_train
        self.mse_val = mse_val
        self.model_params = model_params

    def __lt__(self, other):
        return self.r2_val < other.r2_val

    def __repr__(self):
        return f"{self.__class__.__name__}({self.r2_train=:.5f}, {self.r2_val=:.5f}, {self.mse_train=:.5f}, {self.mse_val=:.5f}, {self.model_params=})"

class FinalResult:
    def __init__(self, r2_test, mse_test, model_params, model_state, true, pred, history):
        self.r2_test = r2_test
        self.mse_test = mse_test
        self.model_params = model_params
        self.model_state = model_state
        self.true = true
        self.pred = pred
        self.history = history

    def __repr__(self):
        return f"{self.__class__.__name__}({self.r2_test=:.5f}, {self.mse_test=:.5f}, {self.model_params=})"
