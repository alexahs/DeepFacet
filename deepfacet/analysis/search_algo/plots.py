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




class Plots:

    def __init__(self, model_gens_root_dir = "../../dev/nets/model_gens_9_pore", num_gens = 6):
        # assert(search_type in ("target", "strongest", "strongest_with_syms", "weakest", "weakest_with_syms"))
        # self.search_type = search_type

        # if search_type == "target":
            # type_path = "9.4_design"
        # else:
            # type_path = search_type

        # self.data_path = Path(model_gens_root_dir) / type_path
        self.model_gens_root_dir = Path(model_gens_root_dir)
        self.num_gens = num_gens

    def gen_dir(self, search_type, gen):
        if search_type == "target":
            search_type = "9.4_design"
        return self.model_gens_root_dir / search_type / f"0{gen}_gen"

    def print_best_yield_stress(self):

        for s in ["strongest", "strongest_with_syms", "weakest", "weakest_with_syms"]:
            ys_list = []
            b_list = []
            p_list = []
            for gen in range(6):
                ys, batch, pore_num = self.get_max_yield_stress(s, gen)
                ys_list.append(ys)
                b_list.append(batch)
                p_list.append(pore_num)
            print(f"{s}:\"best\":\n{ys_list}\n{b_list}\n{p_list}")

    def get_max_yield_stress(self, search_type, gen):

        assert("strongest" in search_type or "weakest" in search_type)
        data = np.load(self.gen_dir(search_type, gen) / "screened_logs.npz", allow_pickle=True)['arr_0'][()]

        yields = []
        pore_nums = []
        batches = []

        for run, d in data.items():
            yields.append(d["mean_yield"])
            meta = d["meta"]
            pore_nums.append(meta["pore_num"])
            batches.append(meta["batch"])

        if "strongest" in search_type:
            idx = np.argmax(yields)
        elif "weakest" in search_type:
            idx = np.argmin(yields)

        return yields[idx], batches[idx], pore_nums[idx]

    @staticmethod
    def r2_score(pred, true):
        if len(pred) == 1:
            return 0
        mean = np.mean(true)
        SS_res = np.sum((true - pred)**2)
        SS_tot = np.sum((true - mean)**2)
        r2 = (1 - SS_res/SS_tot)
        return r2

    def yield_stress(self, search_type):
        results = {
            "strongest": {
                "mu_simulated":  [9.989, 9.998, 9.845, 9.967 , 10.028, 9.978],
                "std_simulated": [0.193, 0.066, 0.232, 0.095 , 0.171 , 0.093],
                "mu_target":     [9.984, 10.11, 9.926, 10.019, 9.970 , 10.071],
                "std_target":    [0.021, 0.021, 0.021, 0.035 , 0.021 , 0.018],
                "best":          [10.1709544106338, 10.12869054275232, 10.126514862347829, 10.130310696049921, 10.345948372676471, 10.10288931303243]},
            "weakest": {
                "mu_simulated":  [8.617, 8.625, 8.647, 8.639, 8.616, 8.647],
                "std_simulated": [0.095, 0.106, 0.080, 0.080, 0.066, 0.085],
                "mu_target":     [8.292, 8.392, 8.434, 8.483, 8.611, 8.517],
                "std_target":    [0.022, 0.036, 0.021, 0.017, 0.011, 0.018],
                "best":          [8.402497608692292, 8.354529101128332, 8.484177963963804, 8.508791226878152, 8.457529503660439, 8.50957041691759]},
            "target": {
                "mu_simulated":  [9.326, 9.492, 9.434, 9.376, 9.495, 9.340],
                "std_simulated": [0.212, 0.197, 0.141, 0.133, 0.179, 0.204],
                "mu_target":     [9.400, 9.400, 9.400, 9.400, 9.400, 9.400],
                "std_target":    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]},
            "strongest_with_syms": {
                "mu_simulated":  [10.690, 10.680, 10.634, 10.640, 10.063, 10.074],
                "std_simulated": [0.102 , 0.070 , 0.106 , 0.063 , 0.247 , 0.25  ],
                "mu_target":     [10.001, 10.647, 10.672, 10.614, 10.346, 10.178],
                "std_target":    [0.040 , 0.038 , 0.022 , 0.040 , 0.036 , 0.005 ],
                "best":          [10.911180839215039, 10.77665525037896, 10.748456707978315, 10.768488850985694, 10.749689130528623, 10.710424645504862]},
            "weakest_with_syms": {
                "mu_simulated":  [8.466, 8.597, 8.526, 8.653, 8.581, 8.496],
                "std_simulated": [0.057, 0.047, 0.075, 0.072, 0.150, 0.075],
                "mu_target":     [8.106, 8.286, 8.388, 8.428, 8.429, 8.487],
                "std_target":    [0.115, 0.019, 0.010, 0.013, 0.016, 0.013],
                "best":          [8.36000918502357, 8.507663583452125, 8.395648934642336, 8.534139083743355, 8.30240771991082, 8.315744320810811]}
        }
        return results[search_type]

    def get_best_yield_gen(self, search_type):
        ys = self.yield_stress(search_type)["mu_simulated"]

        if "strongest" in search_type:
            return np.argmax(ys)
        elif "weakest" in search_type:
            return np.argmin(ys)
        elif "target" in search_type:
            return np.argmin((ys-9.4)**2)
        else:
            raise NotImplementedError

    def plot1(self):
        """
        NOT USED
        generation plots of yield stress with syms and no syms (all search types)
        """

        fig = plt.figure(figsize=(4.7747, 3.9509))
        gen = range(self.num_gens)

        gs = GridSpec(3, 1)
        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[2, 0]),
            # fig.add_subplot(gs[0, 1]),
            # fig.add_subplot(gs[1, 1]),
            # fig.add_subplot(gs[2, 1]),
        ]
        gs.update(hspace=0.1)
        styles = ["-", "-"]
        legends = ["reduced design space", "full design space"]

        for i, st in enumerate(["strongest", "strongest_with_syms", "weakest", "weakest_with_syms", "target"]):
            ys = self.yield_stress(st)
            j = i // 2
            yerr = np.array(ys["std_simulated"])**2
            y_sim = np.array(ys["mu_simulated"])
            y_target = np.array(ys["mu_target"])
            # yerr = (y_target - y_sim), (np.maximum(0, -yerr), np.maximum(0, yerr))
            ax[j].errorbar(gen, y_sim, yerr=yerr,
                linestyle=styles[i % 2],
                label=legends[i%2],
                capsize=1)

            # ax[j+3].plot(gen, y_target)

        # plt.tight_layout()
        box = ax[0].get_position()
        ax[0].legend(bbox_to_anchor=(-0.01, 1.3, 1.02, .10), ncol=2, mode="expand")
        ax[1].set_ylabel("True yield stress [GPa]")

        for i in (0, 1):
            ax[i].set_xticklabels([])

        coords = [(0.01, 0.8), (0.01, 0.8), (0.01, 0.8)]
        for i, label in enumerate(("a)", "b)", "c)")):
            ax[i].text(*coords[i], label, transform=ax[i].transAxes)

        ax[2].set_xlabel("Generation")
        # plt.savefig("figs/generations.pdf")

    def plot2(self):
        """
        NOT USED
        generation plots of true and pred yield stress (all search types)
        """


        search_types = ["strongest", "weakest", "target"]
        # search_types = ["strongest_with_syms", "weakest_with_syms"]


        fig = plt.figure(figsize=(4.7747, 3.9509))
        gen = range(self.num_gens)

        gs = GridSpec(3, 1)
        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[2, 0]),
            # fig.add_subplot(gs[0, 1]),
            # fig.add_subplot(gs[1, 1]),
            # fig.add_subplot(gs[2, 1]),
        ]
        gs.update(hspace=0.1)
        styles = ["-", "-"]
        legends = ["true", "predicted"]

        for i, st in enumerate(search_types):
            ys = self.yield_stress(st)
            # j = i // 2
            yerr_sim = np.array(ys["std_simulated"])**2
            yerr_target = np.array(ys["std_target"])**2
            y_sim = np.array(ys["mu_simulated"])
            y_target = np.array(ys["mu_target"])
            # yerr = (y_target - y_sim), (np.maximum(0, -yerr), np.maximum(0, yerr))
            ax[i].errorbar(gen, y_sim, yerr=yerr_sim,
                linestyle=styles[i % 2],
                label=legends[0],
                capsize=1)

            ax[i].errorbar(gen, y_target, yerr=yerr_target,
                linestyle=styles[i % 2],
                label=legends[1],
                capsize=1)

            # ax[j+3].plot(gen, y_target)

        # plt.tight_layout()
        box = ax[0].get_position()
        ax[0].legend(bbox_to_anchor=(-0.01, 1.3, 1.02, .10), ncol=2, mode="expand")
        ax[1].set_ylabel("Yield stress [GPa]")

        for i in (0, 1):
            ax[i].set_xticklabels([])

        coords = [(0.01, 0.8), (0.01, 0.8), (0.01, 0.8)]
        for i, label in enumerate(("a)", "b)", "c)")):
            ax[i].text(*coords[i], label, transform=ax[i].transAxes)

        ax[2].set_xlabel("Generation")

    def plot3(self, search_type):
        """generation plots of yield stress with syms and no syms (single search type, no target)"""

        gen = range(self.num_gens)

        fig, ax = plt.subplots(nrows = 2, sharey=True)#, figsize = (4.7747/2, 2.9509))

        styles = ["-", "-"]
        legends = ["true", "predicted"]

        for i, st in enumerate([search_type, f"{search_type}_with_syms"]):
            ys = self.yield_stress(st)

            yerr_sim = np.array(ys["std_simulated"])
            yerr_target = np.array(ys["std_target"])**2
            print(i, search_type, f"{yerr_sim=}", f"{np.mean(yerr_sim)=:.3f}")

            y_sim = np.array(ys["mu_simulated"])
            # y_sim_best = np.array(ys["best"])
            y_target = np.array(ys["mu_target"])
            # yerr = (y_target - y_sim), (np.maximum(0, -yerr), np.maximum(0, yerr))
            # ax[i].errorbar(gen, y_sim, yerr=yerr_sim,
            #     linestyle=styles[i % 2],
            #     label="true",
            #     capsize=1)
            # ax[i].plot(y_sim, "o-", label="true", markersize=2)
            # ax[i].plot(y_sim, "o-", label="mean", markersize=2)


            ax[i].errorbar(gen, y_sim, yerr=yerr_sim,
                linestyle="-",
                label="true",
                capsize=1)
            ax[i].plot(y_target, "o-", label="predicted", markersize=2)
            # ax[i].plot(gen, y_target)

        # plt.tight_layout()
        box = ax[0].get_position()
        if "strongest" in search_type:
            loc = 1
            # ax[0].set_ylim([9.6, 11])
            # ax[1].set_ylim([9.6, 11])
        else:
            loc = 4
        ax[0].legend(loc=loc)
        ax[0].set_xticklabels([])

        # coords = [(0.7, 0.8), (0.76, 0.8)] ("Reduced design space", "Full design space")
        coords = [(0.01, 0.8), (0.01, 0.8)]
        for i, label in enumerate(("a)", "b)")):
            ax[i].text(*coords[i], label, transform=ax[i].transAxes)
            # ax[i].set_ylabel(r"$\sigma_{\text{yield}}$ [GPa]")
            ax[i].set_ylabel("Yield\nstress [GPa]")
            # ax[i].legend()

        ax[1].set_xlabel("Number of generations")
        fig.align_ylabels(ax)
        plt.tight_layout()

        plt.savefig(f"figs/generations_{search_type}_and_syms.pdf")
        return ax

    def plot4(self):
        """generation plots of yield stress target 9.4 GPa"""
        gen = range(self.num_gens)
        fig, ax = plt.subplots(nrows = 1, figsize = (4.7747, 2.9509/3*2))
        st = "target"

        ys = self.yield_stress(st)
        yerr_sim = np.array(ys["std_simulated"])
        yerr_target = np.array(ys["std_target"])**2

        y_sim = np.array(ys["mu_simulated"])
        y_target = np.array(ys["mu_target"])
        ax.errorbar(gen, y_sim, yerr=yerr_sim,
            label="true",
            capsize=1)

        # ax.plot(gen, y_target, "o-", markersize=2, label="predicted")
        ax.plot([0, 5], [9.4, 9.4], "k--", label="target")
        # ax.errorbar(gen, y_target, yerr=yerr_target,
        #     label="predicted",
        #     capsize=1)

        ax.legend()
        ax.set_ylabel("Yield stress [GPa]")
        ax.set_xlabel("Number of generations")
        ax.set_ylim([9.1, 9.8])
        plt.tight_layout()
        plt.savefig(f"figs/generations_target.pdf")

    def plot5(self, search_type, gen):
        """
        NOT USED
        histogram per type
        """


        from scipy.stats import norm
        yield_full = []
        yield_screened = []

        data_full = np.load(self.gen_dir(search_type, gen) / "base_logs.npz", allow_pickle=True)['arr_0'][()]
        preds = []
        for g in range(6):
            data_screened = np.load(self.gen_dir(search_type, g) / "screened_logs.npz", allow_pickle=True)['arr_0'][()]
            for d in data_screened.values():
                yield_screened.append(d['mean_yield'])
                predicted = self.yield_stress(search_type)["mu_target"][g]
                preds.append(predicted)


        predicted = np.mean(preds)

        for d in data_full.values():
            yield_full.append(d['mean_yield'])

        # for d in data_screened.values():
        mu_sim = np.mean(yield_screened)
        std_sim = np.std(yield_screened)
        mu_pred = np.mean(predicted)
        std_pred = np.std(predicted)


        fig, ax = plt.subplots()
        colors = ["tab:blue", "tab:red", "tab:green"]
        labels = ["Randomly selected",
                  r"Simulated, $\overline{\sigma} = $" + f"{mu_sim:.3f}" + r"$\pm$" + f"{std_sim:.3f}"]
                  # r"Target, $\overline{\sigma}$ = " + f"{mu_pred:.3f}" + r"$\pm$" + f"{std_pred:.3f}"]
        alpha = 0.5
        bins = np.linspace(8.3, 12, 50)
        xmin = np.inf
        xmax = 0
        for i, stress in enumerate([yield_full, yield_screened]):

            counts, bins = np.histogram(stress, bins=bins)
            idx = np.where(counts != 0)[0]
            if bins[idx[0]] < xmin:
                xmin = bins[idx[0]]
            if bins[idx[-1]] > xmax:
                xmax = bins[idx[-1]]

            ax.hist(bins[:-1], bins, density=False, weights=counts, color=colors[i], alpha=alpha, label=labels[i])

        if search_type == "target":
            line = Line2D([predicted, predicted], [0, 10], color=colors[2], alpha=alpha, linewidth = 1, label="target value")
            ax.add_line(line)


        xvals = np.linspace(mu_sim - 3*std_sim, mu_sim + 3*std_sim)

        ax.set_xlabel("Yield Stress [GPa]")
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_xlim([xmin-0.5, xmax+0.5])
        search_type = search_type.replace("_", " ")
        ax.set_title(f"{search_type}" + str(gen))

        plt.tight_layout()

    def plot6(self, search_type):
        """histogram with and without syms (strongest/weakest)"""

        from scipy.stats import norm
        yield_full = []
        yield_screened_sym = []
        yield_screened_no_sym = []

        for g in range(6):
            data_screened = np.load(self.gen_dir(search_type, g) / "screened_logs.npz", allow_pickle=True)['arr_0'][()]
            data_screened_sym = np.load(self.gen_dir(f"{search_type}_with_syms", g) / "screened_logs.npz", allow_pickle=True)['arr_0'][()]
            for d in data_screened.values():
                yield_screened_no_sym.append(d['mean_yield'])
            for d in data_screened_sym.values():
                yield_screened_sym.append(d['mean_yield'])


        data_full = np.load(self.gen_dir(search_type, 0) / "base_logs.npz", allow_pickle=True)['arr_0'][()]
        for d in data_full.values():
            yield_full.append(d['mean_yield'])

        # for d in data_screened.values():
        # mu_sim = np.mean(yield_screened)
        # std_sim = np.std(yield_screened)
        # mu_pred = np.mean(predicted)
        # std_pred = np.std(predicted)



        fig, ax = plt.subplots()
        colors = ["tab:blue", "tab:green", "tab:red"]
        # labels = ["Randomly selected",
        #           r"Simulated, $\overline{\sigma} = $" + f"{mu_sim:.3f}" + r"$\pm$" + f"{std_sim:.3f}"]
                  # r"Target, $\overline{\sigma}$ = " + f"{mu_pred:.3f}" + r"$\pm$" + f"{std_pred:.3f}"]
        labels = ["random search", "ML search (reduced space)", "ML search (full space)"]
        alpha = 0.6
        bins = np.linspace(8.3, 12, 60)
        xmin = np.inf
        xmax = 0
        for i, stress in enumerate([yield_full, yield_screened_no_sym, yield_screened_sym]):

            counts, bins = np.histogram(stress, bins=bins)
            idx = np.where(counts != 0)[0]
            if bins[idx[0]] < xmin:
                xmin = bins[idx[0]]
            if bins[idx[-1]] > xmax:
                xmax = bins[idx[-1]]

            ax.hist(bins[:-1],
                    bins,
                    density=False,
                    weights=counts,
                    alpha=alpha,
                    label=labels[i],
                    color=colors[i],
                    edgecolor=colors[i],
                    # edgecolor="black",
                    histtype= "stepfilled"
                    )

        xmax += 0.25
        xmin -= 0.25
        if search_type == "weakest":
            xmax += 0.1
        # xvals = np.linspace(mu_sim - 3*std_sim, mu_sim + 3*std_sim)



        ax.text(0.05, 0.9, "c)", transform=ax.transAxes)


        ax.set_xlabel("True yield stress [GPa]")
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, 30])
        # search_type = search_type.replace("_", " ")
        # ax.set_title(f"{search_type}")

        plt.tight_layout()

        plt.savefig(f"figs/hist_{search_type}.pdf")

    def plot7(self):
        """histogram target"""

        search_type = "target"
        yield_full = []
        yield_screened = []

        for g in range(6):
            data_screened = np.load(self.gen_dir(search_type, g) / "screened_logs.npz", allow_pickle=True)['arr_0'][()]
            for d in data_screened.values():
                yield_screened.append(d['mean_yield'])


        data_full = np.load(self.gen_dir(search_type, 0) / "base_logs.npz", allow_pickle=True)['arr_0'][()]
        for d in data_full.values():
            yield_full.append(d['mean_yield'])


        fig, ax = plt.subplots()
        colors = ["tab:blue", "tab:red", "tab:orange"]
        labels = ["random search", "ML search"]
        alpha = 0.6
        bins = np.linspace(8.3, 12, 60)
        xmin = np.inf
        xmax = 0
        for i, stress in enumerate([yield_full, yield_screened]):

            counts, bins = np.histogram(stress, bins=bins)
            idx = np.where(counts != 0)[0]
            if bins[idx[0]] < xmin:
                xmin = bins[idx[0]]
            if bins[idx[-1]] > xmax:
                xmax = bins[idx[-1]]

            ax.hist(bins[:-1],
                    bins,
                    density=False,
                    weights=counts,
                    alpha=alpha,
                    label=labels[i],
                    color=colors[i],
                    edgecolor=colors[i],
                    # edgecolor="black",
                    histtype= "stepfilled"
                    )

        line = Line2D([9.4, 9.4], [0, 20], linestyle = "--", color="black", alpha=1, linewidth = 1, label="target value")
        ax.add_line(line)


        xmax += 0.25
        xmin -= 0.25
        if search_type == "weakest":
            xmax += 0.1
        # xvals = np.linspace(mu_sim - 3*std_sim, mu_sim + 3*std_sim)

        ax.set_xlabel("True yield stress [GPa]")
        ax.legend(loc=2)
        ax.set_ylabel("Counts")
        ax.set_xlim([xmin, xmax])
        # ax.set_ylim([0, 30])
        # search_type = search_type.replace("_", " ")
        # ax.set_title(f"{search_type}")

        plt.tight_layout()

        plt.savefig(f"figs/hist_{search_type}.pdf")
        
    def get_distribution_widths(self, search_type):
        """histogram with and without syms (strongest/weakest)"""

        from scipy.stats import norm

        target_yields = []
        for g in range(6):
            data_target = np.load(f"data/9.4_design/0{g}_gen/screened_logs.npz", allow_pickle=True)['arr_0'][()]
            for d in data_target.values():
                target_yields.append(d['mean_yield'])

        mu = np.mean(target_yields)
        std = np.std(target_yields)
        
        yield_full = []

        data_full = np.load(f"data/strongest/05_gen/base_logs.npz", allow_pickle=True)['arr_0'][()]
        for d in data_full.values():
            yield_full.append(d['mean_yield'])

        print(f"range = {np.max(yield_full) - np.min(yield_full)}")

        print(f"{mu=:.4f}, {std=:.4f}")
        exit()
        
        yield_full = []

        data_full = np.load(f"data/strongest/05_gen/base_logs.npz", allow_pickle=True)['arr_0'][()]
        for d in data_full.values():
            yield_full.append(d['mean_yield'])
            print(d)



        mu = np.mean(yield_full)
        std = np.std(yield_full)

        print(f"{mu=:.4f}, {std=:.4f}")
        exit()

        

        # for d in data_screened.values():
        # mu_sim = np.mean(yield_screened)
        # std_sim = np.std(yield_screened)
        # mu_pred = np.mean(predicted)
        # std_pred = np.std(predicted)



        fig, ax = plt.subplots()
        colors = ["tab:blue", "tab:green", "tab:red"]
        # labels = ["Randomly selected",
        #           r"Simulated, $\overline{\sigma} = $" + f"{mu_sim:.3f}" + r"$\pm$" + f"{std_sim:.3f}"]
                  # r"Target, $\overline{\sigma}$ = " + f"{mu_pred:.3f}" + r"$\pm$" + f"{std_pred:.3f}"]
        labels = ["random search", "ML search (reduced space)", "ML search (full space)"]
        alpha = 0.6
        bins = np.linspace(8.3, 12, 60)
        xmin = np.inf
        xmax = 0
        for i, stress in enumerate([yield_full, yield_screened_no_sym, yield_screened_sym]):

            counts, bins = np.histogram(stress, bins=bins)
            idx = np.where(counts != 0)[0]
            if bins[idx[0]] < xmin:
                xmin = bins[idx[0]]
            if bins[idx[-1]] > xmax:
                xmax = bins[idx[-1]]

            ax.hist(bins[:-1],
                    bins,
                    density=False,
                    weights=counts,
                    alpha=alpha,
                    label=labels[i],
                    color=colors[i],
                    edgecolor=colors[i],
                    # edgecolor="black",
                    histtype= "stepfilled"
                    )

        xmax += 0.25
        xmin -= 0.25
        if search_type == "weakest":
            xmax += 0.1
        # xvals = np.linspace(mu_sim - 3*std_sim, mu_sim + 3*std_sim)



        ax.text(0.05, 0.9, "c)", transform=ax.transAxes)


        ax.set_xlabel("True yield stress [GPa]")
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, 30])
        # search_type = search_type.replace("_", " ")
        # ax.set_title(f"{search_type}")

        plt.tight_layout()

        plt.savefig(f"figs/hist_{search_type}.pdf")

def r2_score(pred, true, debug=False):
    if len(pred) == 1:
        return 0
    mean = np.mean(true)
    SS_res = np.sum((true - pred)**2)
    SS_tot = np.sum((true - mean)**2)
    r2 = (1 - SS_res/SS_tot)

    return r2

def plot_r2_in_gens():
    search_type = "weakest" 
    r2_scores = []
    rmse = []
    f_gen = 5

    for gen in range(0, 6):
        true, pred = np.load(f"data/{search_type}/0{gen}_gen/true_vs_pred_test.npy", allow_pickle=True)
        r2 = r2_score(pred, true)
        r2_scores.append(r2)
        mse = np.sum((true - pred)**2)/len(true)
        rmse.append(np.sqrt(mse))
        if gen == f_gen:
            best_true = true
            best_pred = pred
        print(f"gen {gen}, {r2=}")
        

    fig = plt.figure(figsize=(4.7747, 3.9509))
    #fig = plt.figure(figsize=(4.7747, 2.9509))
    gs = GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1:])
    
    
    #ax1.plot(list(range(0, 6)), rmse, "ro--", markersize=2)
    ax1.plot(list(range(0, 6)), r2_scores, "ro--", markersize=2)
    if search_type == "9.4_design":
        ax1.set_ylim([0.4, 0.7])
        ax1.set_yticks([0.4, 0.55, 0.7])
    else:
        ax1.set_ylim([0.5, 1.0])
    ax1.set_ylabel(r"$R^2$ score")
    ax1.set_xlabel("Generation")
    plot_gen_performance(ax2, best_true, best_pred)

    plt.tight_layout()
    gs.update(hspace=0.65)
    plt.savefig(f"figs/{search_type}_gen_r2.pdf")



def plot_gen_performance(ax, true, pred):

    #md precision
    std = 0.11189394170643414

    #true, pred = np.load("data/gen5_true_vs_pred_test.npy", allow_pickle=True)


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

def main():
    """
    in use: plots 3, 4 6 and 7
    """
    p = Plots()
    # p.plot3("strongest")
    # p.plot3("weakest")
    # p.plot4()
    # p.plot6("strongest")
    #p.plot6("weakest")
    # p.plot7()

    #p.get_distribution_widths("strongest")
        # break
    plot_r2_in_gens()
    plt.show()



if __name__ == '__main__':
    main()
    # Plots().print_best_yield_stress()
