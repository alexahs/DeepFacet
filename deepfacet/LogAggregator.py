import numpy as np
import lammps_logfile
import os, sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm
plt.style.use('ggplot')

barToGPa = 1e-4
picoToNs = 1e-3


class LogAggregator:

    def __init__(self, raw_data_path, output_path):
        """
        root_dir: parent directory of simulation batches
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.quantities = None

    @staticmethod
    def get_system_pore_binary(atoms_data_path):
        """
        atoms_data_path: atoms.data path
        """
        with open(atoms_data_path) as infile:
            _ = infile.readline()
            line = infile.readline()
            beg, end = line.find("["), line.find("]")
            nums = line[beg+1:end]
            nums = nums.split(",")

        nums = np.array(nums, dtype=np.int) -1
        binary = np.zeros(27, dtype=np.int)
        binary[nums] = 1
        return tuple(binary)

    @staticmethod
    def get_system_num(sim_dir):
        """
        sim_dir: directory of simulation
        """
        beg = sim_dir.find("pore_9_") + 7
        end = sim_dir.find("_T")
        return sim_dir[beg:end]

    @staticmethod
    def get_yield_idx(time, stress, debug=False):
        """
        returns index of yield stress
        """
        return np.argmax(stress)

    @staticmethod
    def _get_yield_depr(time, stress, debug=False):
        peaks = find_peaks(stress, distance=stress.shape[0]/2, prominence=0.6)
        if debug:
            if len(peaks[0]>0):
                plt.plot(time, stress, 'g-')
                plt.scatter(time[peaks[0][0]], stress[peaks[0][0]], marker='x', s=50, c="k", label="peaks yield")
                plt.legend()
                plt.show()
            else:
                plt.plot(time, stress, 'g-')
                idx = np.argmax(stress)
                plt.scatter(time[idx], stress[idx], marker='x', s=50, c="k", label="argmax yield")
                plt.legend()
                plt.show()


        if len(peaks[0]) > 0:
            return peaks[0][0]
        else:
            return np.argmax(stress)

    @staticmethod
    def get_residual_idx(time, stress, time_threshold=0.15, debug=False):
        """
        stress: arr
        time: arr

        returns index of residual stress
        """
        traughs = find_peaks(-stress, prominence=0.1)[0]
        residual = None

        if len(traughs) > 0:
            residual = traughs[0]
            if len(traughs) > 1:
                for i in range(len(traughs)-1):
                    tr0 = traughs[i]
                    tr1 = traughs[i+1]
                    if(time[tr1] - time[tr0]) > time_threshold:
                        residual = tr0
                        break
                    else:
                        residual = tr1

        if debug:
            if residual is not None:
                plt.plot(time, stress, 'g-')
                if len(traughs) > 0:
                    for tr in traughs:
                        plt.scatter(time[tr], stress[tr], marker='o', s=50, label="others")
                plt.scatter(time[residual], stress[residual], marker='x', s=50, c="k", label="residual")
                plt.legend()
                plt.show()
                plt.clf()
            else:
                print("no res")

        return residual

    @staticmethod
    def get_log_data(log_path, quantities:list, run_nums:list, t0=0):
        """
        values: list of str of quantities in log
        run_nums: list of ints of lammps run nums
        """

        try:
            log = lammps_logfile.File(log_path)
        except:
            print("Warning! unable to read ", log_path)
            return None


        values = {}
        for name in quantities:
            tmp_values = []
            for run_num in run_nums:
                # try:
                vals = log.get(name, run_num=run_num)
                if run_num != run_nums[-1]:
                    tmp_values.extend(vals[:-1])
                else:
                    tmp_values.extend(vals)

                # except:
                    # print(f"Warning! run_num {run_num} not in {log_path}")
                    # continue

            values[name] = np.array(tmp_values)
            if name == "Time" and t0 != 0:
                values["Time"] += t0


        return values

    def aggregate(self, fname_out:str, batches:list = None, rerun_t0 = None, savgol_window=31, verbose=False):
        """
        batches: list of str
        rerun_t0: initial time of reruns in ps
        write: str, output log data to npz
        """

        if rerun_t0 is None:
            print("rerun_t0 is None. Using 1600 ps for all batches")
            rerun_t0 = [1600]*len(batches)
        else:
            assert(len(rerun_t0) == len(batches)), "len(rerun_t0) != len(batches)"


        if batches is None:
            batches = os.listdir(self.raw_data_path)

        d = {}
        dt = 0.002*1e-3
        system_count = 0
        for batch, t0 in zip(batches, rerun_t0):
            simulations = os.listdir(os.path.join(self.raw_data_path, batch))
            print(f"Processing {len(simulations)} results in {batch}..")
            for sim_dir in simulations:
                if 'pore_9' not in sim_dir:
                    continue

                run_count = 0
                full_sim_path = os.path.join(self.raw_data_path, batch, sim_dir)
                system_num = LogAggregator.get_system_num(sim_dir)
                system_binary = LogAggregator.get_system_pore_binary(os.path.join(full_sim_path, 'atoms.data'))
                main_log_path = os.path.join(full_sim_path, 'log.lammps')

                d[system_count] = {}
                d[system_count]['runs'] = {}
                d[system_count]['meta'] = {}
                d[system_count]['meta']['batch'] = batch
                d[system_count]['meta']['pore_num'] = system_num
                d[system_count]['meta']['pore_binary'] = system_binary

                vals = LogAggregator.get_log_data(main_log_path, ["Time", "Pzz"], [1, 2], t0=-100.0)
                time = vals["Time"]*picoToNs
                pzz = savgol_filter(vals["Pzz"], window_length=savgol_window, polyorder=2)*barToGPa
                yield_idx = LogAggregator.get_yield_idx(time, pzz)
                residual_idx = LogAggregator.get_residual_idx(time, pzz)

                if residual_idx is not None:
                    if yield_idx > residual_idx:
                        yield_idx = LogAggregator._get_yield_depr(time, pzz)


                yields = []
                residuals = []

                if residual_idx is None:
                    residual_idx = np.argmin(pzz[yield_idx:]) + yield_idx
                    if verbose:
                        print(f"Warning! no residual stress in {batch}_{system_num}, main run. Using min stress val")

                run_dict = {}
                run_dict['Time'] = time
                run_dict['Pzz'] = pzz
                run_dict['yield_idx'] = yield_idx
                run_dict['residual_idx'] = residual_idx
                run_dict['yield_stress'] = pzz[yield_idx]
                run_dict['residual_stress'] = pzz[residual_idx]
                d[system_count]['runs'][run_count] = run_dict

                residuals.append(pzz[residual_idx])
                yields.append(pzz[yield_idx])



                num_reruns = len([i for i in os.listdir(full_sim_path) if 'rerun_' in i])


                for i in range(1, num_reruns+1):
                    rerun_dir = os.path.join(full_sim_path, f'rerun_{i}')
                    if not os.path.exists(rerun_dir):
                        continue

                    rerun_log = os.path.join(rerun_dir, 'log.lammps')
                    vals = LogAggregator.get_log_data(rerun_log, ["Time", "Pzz"], [0], t0=t0)
                    time = vals["Time"]*picoToNs
                    pzz = savgol_filter(vals["Pzz"], window_length=savgol_window, polyorder=2)*barToGPa
                    yield_idx = LogAggregator.get_yield_idx(time, pzz)
                    residual_idx = LogAggregator.get_residual_idx(time, pzz)

                    if residual_idx is None:
                        residual_idx = np.argmin(pzz)
                        if verbose:
                            print(f"Warning! no residual stress in {batch}_{system_num}, rerun {i}. Using min stress val")

                    run_count += 1
                    run_dict = {}
                    run_dict['Time'] = time
                    run_dict['Pzz'] = pzz
                    run_dict['yield_idx'] = yield_idx
                    run_dict['residual_idx'] = residual_idx
                    run_dict['yield_stress'] = pzz[yield_idx]
                    run_dict['residual_stress'] = pzz[residual_idx]
                    d[system_count]['runs'][run_count] = run_dict

                    yields.append(pzz[yield_idx])
                    residuals.append(pzz[residual_idx])


                d[system_count]['mean_yield'] = np.mean(yields)
                d[system_count]['std_yield'] = np.std(yields)
                d[system_count]['mean_residual'] = np.mean(residuals)
                d[system_count]['std_residual'] = np.std(residuals)

                # if np.mean(yields) < 8.6:
                #     print("outlier:", batch, system_num, f"yield stress:{np.mean(yields):.3f} GPa")


                system_count += 1
        self.quantities = d

        out_path = os.path.join(self.output_path, fname_out)


        np.savez(out_path, d, allow_pickle=True)
        return d

    def write_model_ready(self, logs_fname, fname_out):

        data = np.load(os.path.join(self.output_path, logs_fname), allow_pickle=True)['arr_0'][()]
        out_path = os.path.join(self.output_path, fname_out)
        n = len(data)

        X = np.zeros((n, 27+2))

        k = 0
        for key, val in data.items():
            yield_stress = val["mean_yield"]
            residual_stress = val["mean_residual"]
            binary = val["meta"]["pore_binary"]
            X[k,:-2] = binary
            X[k, -1] = yield_stress
            X[k, -2] = residual_stress
            k+=1

        # print(X)
        np.save(out_path, X)
        print(f"wrote {out_path}, {k} samples")

    @staticmethod
    def plot_full_dataset(fname, num_plots=None):
        data = np.load(fname, allow_pickle=True)['arr_0'][()]
        # for run, v in data[0]["runs"].items():
        #     print(run)
        #     print(v)

        if num_plots is None:
            num_plots = len(data)
        else:
            num_plots = min(len(list(data.keys())), num_plots)


        c = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]

        for i in range(num_plots):
            run = data[i]['runs']
            meta = data[i]['meta']
            mean_yield = data[i]['mean_yield']
            #print(meta["pore_num"], mean_yield)
            #continue
            fig, ax = plt.subplots(nrows=1)
            # print(run.items())
            for j,vals in run.items():
                time = vals['Time']
                pzz = vals['Pzz']
                ax.plot(time, pzz)#, color=c[j])
                ax.set_title(meta["pore_num"] + f" {mean_yield:.3f} GPa")
                # ax.legend()
                continue
                ax.scatter(time[vals['residual_idx']], pzz[vals['residual_idx']], c=c[j], marker="x")
                ax.scatter(time[vals['yield_idx']], pzz[vals['yield_idx']], c=c[j], marker="x")
                ax.set_xlim([1.5, 2.7])
                ax.set_xlabel("Time [ns]")
                ax.set_ylabel("Stress [GPa]")
                ax.set_title(f"{meta['pore_num']}, {meta['batch']}, yield stress:{mean_yield:.2f}")

            plt.show()

    def plot_histogram(fname, n_bins=20):
        yields = []
        residuals = []
        std_yield = []
        std_residual = []

        # data = np.load(os.path.join(self.output_path, fname), allow_pickle=True)['arr_0'][()]
        data = np.load(fname, allow_pickle=True)['arr_0'][()]

        for d in data.values():
            yields.append(d['mean_yield'])
            residuals.append(d['mean_residual'])
            std_yield.append(d['std_yield'])
            std_residual.append(d['std_residual'])

        fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(16, 8))
        titles = ["Yield Stress", "Residual Stress"]
        colors = ["tab:red", "tab:blue"]
        for i, val in enumerate([yields, residuals]):
            mu = np.mean(val)
            std = np.std(val)
            counts, bins = np.histogram(val, bins=n_bins)
            ax[i].hist(bins[:-1], bins, density=False, weights=counts, color=colors[i], alpha= 0.5, label=f"$\mu={mu:.2f}$\n$\sigma={std:.2f}$")
            ax[i].set_xlabel("Stress [GPa]")
            ax[i].legend()
            ax[i].set_title(titles[i])

        ax[0].set_ylabel(r"Counts")
        # fig.show()
        plt.show()


if __name__ == '__main__':
    # raw_data_path = "runs/9_pore/high_strain_rate/design/strongest/completed_batches"
    # output_path = "../nets/data"
    # fname_logs = "gen_2_strongest.npz" #aggregated logs
    # fname_processed = "gen_2_strongest.npy" #model ready data

    #raw_data_path = "/home/Alexander/compsci/thesis/dummy_batch"
    raw_data_path = "/home/Alexander/compsci/thesis/reduced_space_sims"
    output_path = "./"
    fname = "out.npz"
    # fname = "/home/alexander/compsci/thesis/dev/nets/model_gens_9_pore/strongest_with_syms/05_gen/screened_logs.npz"

    logs = LogAggregator(raw_data_path, output_path)
    logs.aggregate(fname, batches = ["weakest_05_gen"], rerun_t0 = [1600])
    # fname = "test_00_gen_syms.npz"
    # logs.aggregate(fname, batches = ["00_gen"], rerun_t0 = [1600])
    # fname = "../nets/model_gens_9_pore/strongest_with_syms/00_gen/screened_logs.npz"
    LogAggregator.plot_full_dataset(fname)
    plt.show()
    # fname = "../nets/model_gens_9_pore/strongest_with_syms/00_gen/base_logs.npz"
    # LogAggregator.plot_histogram(fname)

    #9.16 (0.23)

    # data = logs.aggregate(fname_logs, batches=["02_batch"])
    # logs.write_model_ready(fname_logs, fname_processed)

    # logs.plot_full_dataset(fname_logs)
    # logs.plot_histogram(fname_logs, n_bins = 20)
    # print(data)
    # plot_main_runs()


#
