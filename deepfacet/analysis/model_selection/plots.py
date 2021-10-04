import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import os, sys, time, re
import matplotlib as mpl
from pathlib import Path
from dataclasses import dataclass
plt.style.use('../../my_ggplot.mplstyle')
sys.path.append("/home/alexander/compsci/thesis/dev/nets/torch/")
# sys.path.append("/home/alexander/compsci/thesis/dev/nets/torch/dnns")
# sys.path.append("/home/alexander/compsci/thesis/dev/nets/torch/cnns")
#from CVResults import Result, FinalResult

def r2_score(pred, true, debug=False):
    if len(pred) == 1:
        return 0
    mean = np.mean(true)
    SS_res = np.sum((true - pred)**2)
    SS_tot = np.sum((true - mean)**2)
    r2 = (1 - SS_res/SS_tot)

    return r2

class Plotter:
    def __init__(self):
        self.num_batch = 31
        pass


    def plot1(self, model_type="CNN"):
        """
        plot training history from best model

        CNN:
        kernel size: 4
        n_kernels: (32, 64, 128)
        n_dense: 512
        MSE on test: 0.0236
        R2 on test: 0.574

        DNN:
        n_nodes: 16
        n_layers: 64
        MSE on test: 0.0568
        R2 on test: -0.0001
        """
        assert(model_type in ["CNN", "DNN"])

        run = 0
        if model_type == "CNN":
            rdir = Path("/home/alexander/compsci/thesis/dev/nets/torch/cnns/grid_search_data")
        elif model_type == "DNN":
            rdir = Path("/home/alexander/compsci/thesis/dev/nets/torch/dnns/grid_search_data")

        data = np.load(rdir / f"scores_run{run}.npz", allow_pickle=True)['arr_0'][()]
        best_model = data[0]
        print(best_model)

        mse = np.sum((best_model.true - best_model.pred)**2)/len(best_model.true)

        print(f"{mse=}")


        fig = plt.figure()
        gs = GridSpec(2, 1)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])

        train_col = "tab:orange"
        val_col = "tab:blue"

        ax0.semilogy(np.array(best_model.history["loss_train"])/self.num_batch, color=train_col, label="training data")
        ax0.semilogy(best_model.history["loss_val"], color=val_col, label="validation data")
        ax0.legend()
        ax0.set_ylabel("MSE loss")
        ax0.set_xticklabels([])

        ax1.plot(best_model.history["r2_train"], color=train_col, label="training data")
        ax1.plot(best_model.history["r2_val"], color=val_col, label="validation data")
        if model_type == "CNN":
            ax1.set_ylim([-1, 1])
        elif model_type == "DNN":
            ax1.set_ylim([-10, 1])
        ax1.set_ylabel(r"$R^2$ score")
        ax1.set_xlabel("Epoch")
        plt.tight_layout()

        gs.update(hspace=0.1)
        plt.savefig(f"figs/{model_type}_training_history.pdf")

    def plot2(self, model_type="CNN"):
        """
        plot true vs pred(test) best cnn model
        """

        #md precision
        std = 0.11189394170643414

        assert(model_type in ["CNN", "DNN"])


        run = 0
        rdir = Path("/home/alexander/compsci/thesis/dev/nets/torch/cnns/grid_search_data")
        data = np.load(rdir / f"scores_run{run}.npz", allow_pickle=True)['arr_0'][()]
        best_model = data[0]
        print(best_model)

        fig, ax = plt.subplots(1)

        # xmin, xmax = np.min(best_model.true), np.max(best_model.true)
        # ymin, ymax = np.min(best_model.pred), np.max(best_model.pred)
        min, max = np.min(best_model.true), np.max(best_model.true)



        mse = np.sum((best_model.true - best_model.pred)**2)/len(best_model.true)
        print(f"{mse=}")
        print(f"r2={r2_score(best_model.pred, best_model.true)}")

        x0 = np.min(best_model.true)
        x1 = np.max(best_model.true)
        y0 = np.min(best_model.pred)
        y1 = np.max(best_model.pred)

        ax.plot((x0, x1), (x0+std, x1+std), "k--", label="MD precision")
        ax.plot((x0, x1), (x0-std, x1-std), "k--")

        ax.plot([min, max], [min, max], "r-", alpha=1, label="ideal model")
        ax.plot(best_model.true, best_model.pred, "bo", alpha=0.5, markersize=2)
        ax.set_xlabel("True yield stress [GPa]")
        ax.set_ylabel("Predicted yield stress [GPa]")
        ax.legend()
        plt.tight_layout()
        plt.savefig("figs/true_vs_pred_cnn_test.pdf")

    def plot3(self, model_type="CNN"):
        """
        plot results from grids search
        """


        if model_type == "CNN":
            rdir = Path("/home/alexander/compsci/thesis/dev/nets/torch/cnns/grid_search_data")
        elif model_type == "DNN":
            rdir = Path("/home/alexander/compsci/thesis/dev/nets/torch/dnns/grid_search_data")

        data = np.load(rdir / f"scores_run0.npz", allow_pickle=True)['arr_0'][()]
        best_model = data[0]

        n_dense_list = 2**np.arange(2, 11)
        n_kernels_list = [(8, 16, 32), (16, 32, 64), (32, 64, 128)]

        data_sorted = [[],[],[]]
        kernel_sizes = [3, 4, 5]
        for n_dense in n_dense_list:
            for i, n_kernels in enumerate(n_kernels_list):
                for model in data[1:]:
                    p = model.model_params
                    if p["kernel_sizes"] == 4 and p["n_kernels"] == n_kernels and p["n_dense"] == n_dense:
                        # print(p)
                        data_sorted[i].append(model)



        for model in data[1:]:
            if model.model_params == {'kernel_sizes': 4, 'bias': True, 'n_kernels': (32, 64, 128), 'n_dense': 512, 'learning_rate': 1e-05, 'batch_size': 32}:
                print(model)

        print(data[0])

        fig = plt.figure()
        gs = GridSpec(2, 1)
        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        ax[0].plot([3], [0.2], color="white", label="No. of kernels", lw = 0)
        for i in range(len(data_sorted)):
            new_dense = []
            mse = []
            r2 = []
            for n_dense in n_dense_list:
                for m in data_sorted[i]:
                    if m.model_params["n_dense"] == n_dense:
                        mse.append(m.mse_val)
                        r2.append(m.r2_val)
                        new_dense.append(n_dense)

            idx = np.argsort(new_dense)
            mse = np.array(mse)[idx]
            r2 = np.array(r2)[idx]
            label = ""
            for k in n_kernels_list[i]:
                label += str(k) + ", "
            label = label[:-2]
            ax[0].plot(np.log2(n_dense_list), mse, "o-", label=label)
            ax[1].plot(np.log2(n_dense_list), r2, "o-")

        # ax[0].legend(loc=1)
        # ax[0].plot(np.log2(n_dense_list), [0.11]*len(n_dense_list))
        ax[0].set_ylabel("RMSE [GPa]")
        ax[1].set_ylabel(r"$R^2$ score")
        ax[1].set_xlabel(r"Number of neurons in dense layer [log$_2$]")
        ax[0].set_xticklabels([])
        # ax[0].set_ylim([0.35, 0.55])
        plt.tight_layout()
        gs.update(hspace=0.1)

        for ax_ in ax:
            box = ax_.get_position()
            ax_.set_position([box.x0, box.y0, box.width*0.72, box.height])

        ax[0].legend(bbox_to_anchor=(1, 0, 0.45, 1.05), mode='expand')
        plt.savefig("figs/grid_search_cnn.pdf")





def get_md_precision():
    rdir = Path("/home/alexander/compsci/thesis/dev/nets/model_gens_9_pore/strongest/00_gen/base_logs.npz")
    data = np.load(rdir, allow_pickle=True)["arr_0"][()]
    std = 0
    mae = 0
    for run, d in data.items():
        std += d["std_yield"]

    std /= len(data.keys())
    print(f"{std=}")




def main():
    p = Plotter()
    # p.plot1("CNN")
    # p.plot1("DNN")
    # p.plot2("CNN")
    p.plot3("CNN")
    plt.show()

def plot_r2_in_gens():
    search_type = "strongest" 
    r2_scores = []
    rmse = []

    for gen in range(0, 6):
        true, pred = np.load(f"data/{search_type}/0{gen}_gen/true_vs_pred_test.npy", allow_pickle=True)
        r2 = r2_score(pred, true)
        r2_scores.append(r2)
        mse = np.sum((true - pred)**2)/len(true)
        rmse.append(np.sqrt(mse))
        print(f"gen {gen}, {r2=}")
        

    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    
    ax1.plot(list(range(0, 6)), r2_scores, "ro--", markersize=2)
    ax1.set_ylim([0.5, 1])
    ax1.set_ylabel(r"$R^2$ score")
    ax1.set_xlabel("Generation")
    plot_gen_performance(ax2, true, pred)

    plt.tight_layout()
    gs.update(hspace=0.5)
    plt.savefig(f"figs/{search_type}_gen_r2.pdf")
    plt.show()



def plot_gen_performance(ax, true, pred):

    #md precision
    std = 0.11189394170643414



    min, max = np.min(true), np.max(true)


    mse = np.sum((true - pred)**2)/len(true)
    print(f"{mse=}")
    print(f"RMSE={np.sqrt(mse)}")
    print(f"r2={r2_score(pred, true)}")

    x0 = np.min(true)
    x1 = np.max(true)
    y0 = np.min(pred)
    y1 = np.max(pred)

    ax.plot((x0, x1), (x0+std, x1+std), "k--", label="MD precision")
    ax.plot((x0, x1), (x0-std, x1-std), "k--")

    ax.plot([min, max], [min, max], "r-", alpha=1, label="ideal model")
    ax.plot(true, pred, "bo", alpha=0.5, markersize=2)
    ax.set_xlabel("True yield stress [GPa]")
    ax.set_ylabel("Predicted yield\nstress [GPa]")
    ax.legend()
    #plt.tight_layout()
    #plt.savefig("figs/gen5_model_true_vs_pred.pdf")
    #plt.show()



if __name__ == '__main__':
    #main()
    # get_md_precision()







    plot_r2_in_gens()








