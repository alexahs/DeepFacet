import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import lammps_logfile
import numpy as np
import os, sys, time, re
import scipy.signal
import scipy.integrate
import scipy.interpolate
import matplotlib as mpl
from pathlib import Path
from dataclasses import dataclass
sys.path.append('../force_of_crystallization')
from fitting import poly_fitter, curve_fitter
plt.style.use('../../my_ggplot.mplstyle')


barToGPa = 1e-4
picoToNs = 1e-3
AngToNano = 1e-1
n_atoms = 12457
k_b = 8.617333262e-5 #eV/K

class WriteData:
    """
    writes thermo output from all simulations to a single npz file
    unit conversions:
    * Ang -> nm
    * ps  -> ns
    * bar -> GPa
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.set_keywords()
        self.conversion_factors = {
            "Pzz": barToGPa,
            "Time": picoToNs,
            "Lz": AngToNano
            }

    def set_keywords(self, keywords=None):
        if keywords:
            if not hasattr(keywords, '__len__'):
                keywords = [keywords]
            self.keywords = keywords
        else:
            self.keywords = ["Time", "Pzz", "Lz"]

    @staticmethod
    def compute_strain(L):
        assert(hasattr(L, '__len__'))
        return abs((L[0]-L)/L[0])

    @staticmethod
    def compute_strain_rate(simulation_time, deformation):
        assert(deformation > 0 and deformation < 1 and simulation_time > 0)
        return (1-deformation)/simulation_time


    def get_thermo_values(self, log_path, run_nums, exclude_keywords=None):
        if run_nums is None:
            run_nums = [0]
        log = lammps_logfile.File(log_path)
        values = {}
        for run_num in run_nums:
            for kw in self.keywords:
                vals = log.get(kw, run_num = run_num)
                factor = self.conversion_factors.get(kw) or 1
                if kw in values:
                    values[kw] = np.append(values[kw], vals*factor)
                else:
                    values[kw] = vals*factor

        return values

    @staticmethod
    def parse_fname(s):
        try:
            T, t, deform =  re.findall(r"[-+]?\d*\.\d+|\d+", str(s))
        except:
            print(f"{s}: path must contain temperature, simulation time and deformation scale")
            raise ValueError

        return int(T), float(t), float(deform)

    def write(self, output_fname, exclude_temps = [1], exclude_times = None, run_nums=None):
        if run_nums is None:
            run_nums = [0]
        exclude_times = exclude_times or []
        all_data = {}
        for sim in self.root_dir.iterdir():
            if not sim.is_dir() or not "scale" in sim.stem:
                continue
            T, t, deform = self.parse_fname(sim)
            if T in exclude_temps:
                continue
            if t in exclude_times:
                continue
            values = self.get_thermo_values(sim / "log.lammps", run_nums=run_nums)
            values["Erate"] = self.compute_strain_rate(t, deform)
            values["Strain"] = self.compute_strain(values["Lz"])
            values["Vol"] = self.load_pore_volume(sim / "ovito_data" / "vol_area_data.npy", values["Strain"])
            all_data[(T, t)] = values

        np.savez(output_fname, all_data)

    def load_pore_volume(self, path, strain):
        vol = np.load(path)[:,1]
        tmp_x = np.linspace(strain[0], strain[-1], vol.shape[0])
        f_vol = scipy.interpolate.interp1d(tmp_x, vol)
        vol = f_vol(strain)*self.conversion_factors["Lz"]**3
        return vol


class SimulationData:
    def __init__(self, values: dict, T:int, final_time:float, window:int = 1):
        self.final_time = final_time
        self.T = T
        required_vals = ["Time", "Erate", "Pzz", "Strain"]
        for v in required_vals:
            assert(v in values)

        self.t = values["Time"]
        self.strain = values["Strain"]
        self.erate = values["Erate"]
        self.pzz = values["Pzz"]
        if "Vol" in values:
            self.vol = values["Vol"]
        if "PotEng" in values:
            self.poteng = values["PotEng"]
        smooth_window = int(self.final_time*window)
        if smooth_window % 2 == 0:
            smooth_window += 1
        self.smooth_window = smooth_window
        self.vol_is_smooth = False
        self.pzz_is_smooth = False
        self.pzz_raw = self.pzz

    def __repr__(self):
        return f"SimulationData({self.T} K, {self.final_time} ns, {self.erate:.3f} 1/ps)"

    def __lt__(self, other):
        return self.T < other.T

    @property
    def tau(self):
        self.find_peaks()
        return self.t[self.peaks[0]]*self.erate

    def smooth(self, x, N=None):
        N = N or self.smooth_window
        return scipy.signal.savgol_filter(x, polyorder=2, window_length=N)

    def smooth_pzz(self, N=None):
        N = N or self.smooth_window
        self.pzz_is_smooth = True
        self.pzz = self.smooth(self.pzz, N)

    def smooth_vol(self, N=None):
        N = N or self.smooth_window
        self.vol_is_smooth = True
        self.vol = self.smooth(self.vol, N)


    def find_peaks(self):
        assert(self.pzz_is_smooth)
        if hasattr(self, 'peaks'):
            return
        peak_distance = round(len(self.pzz))/20
        peaks, properties = scipy.signal.find_peaks(self.pzz, prominence=0.5, distance=peak_distance, height=6)
        self.peaks = peaks

    def find_traughs(self):
        assert(self.pzz_is_smooth)
        if hasattr(self, 'traughs'):
            return
        peak_distance = round(len(self.pzz))/20
        traughs, properties = scipy.signal.find_peaks(-self.pzz, prominence=0.8, distance=peak_distance)
        self.traughs = traughs

    @staticmethod
    def find_nearest_sorted(arr,val):
        idx = np.searchsorted(arr, val, side="left")
        if idx > 0 and (idx == len(arr) or np.fabs(val - arr[idx-1]) < np.fabs(val - arr[idx])):
            return idx-1
        else:
            return idx

    def compute_youngs_modulus(self):
        max_strain = 0.01
        idx = self.find_nearest_sorted(self.strain, max_strain)
        p, cov = np.polyfit(self.strain[:idx], self.pzz[:idx], deg=1, cov=True)
        self.youngs_modulus = p[0]
        self.youngs_modulus_err = np.sqrt(cov[0,0])
        self.stress_fit_x = [0, max_strain]
        self.stress_fit_y = np.polyval(p, self.stress_fit_x)


    @property
    def yield_stress(self):
        self.find_peaks()
        return self.pzz[self.peaks[0]]

    @property
    def yield_strain(self):
        self.find_peaks()
        self.find_traughs()
        d = (self.peaks[0] + self.traughs[0]) // 2
        return self.strain[d]



    def compute_dV(self, num_intervals=10, debug=False):

        self.smooth_pzz()
        self.find_peaks()
        self.find_traughs()


        all_indices = []
        all_fits = []

        peak = self.peaks[0]
        traugh = 0

        len_full = self.strain[traugh:peak].shape[0]
        len_interval = len_full // num_intervals

        len_remain = len_full % num_intervals
        min_strain = self.strain[traugh]
        max_strain = self.strain[peak]

        concat_remain = True if (len_remain/len_interval < 0.2 or len_remain < 3) else False

        interval_inds = []
        all_inds = np.arange(traugh, peak)
        for k in range(num_intervals):
            beg = k*len_interval
            end = beg + len_interval
            if k == (num_intervals - 1) and concat_remain:
                end += len_remain
            interval_inds.append(all_inds[beg:end])
            all_indices.append(all_inds[beg:end])

        dV = []
        dE = []
        prox = []

        for k, inds in enumerate(interval_inds):
            fit_dV = poly_fitter(self.strain[inds], self.vol[inds])
            fit_dE = poly_fitter(self.strain[inds], self.pzz[inds])
            dV.append(fit_dV["slope"])
            dE.append(fit_dE["slope"])
            all_fits.append(np.polyval((fit_dV["slope"], fit_dV["const"]), self.strain[inds]))

            tmp_strain = (self.strain[inds[0]] + self.strain[inds[-1]])/2
            prox.append(tmp_strain / max_strain)


        if debug:
            fig, (ax, ax2, ax3) = plt.subplots(nrows=3)
            for inds, fits in zip(all_indices, all_fits):
                ax.plot(self.strain[inds], self.vol[inds])
                ax.plot(self.strain[inds], fits, 'k--')

            ax2.plot(self.strain[:all_indices[-1][-1]], self.pzz[:all_indices[-1][-1]], 'r-')
            ax3.plot(self.strain[:all_indices[-1][-1]], self.vol[:all_indices[-1][-1]], 'r-')
            ax.set_title(str(self))


        return np.array(dE), np.array(dV), np.array(prox)


class Plotter:
    def __init__(self, log_fname:str):
        self.data = self.load_data(log_fname)

    @staticmethod
    def get_cb(fig, ax, vals=None, orientation="vertical", pad=0.1, cmap = 'RdYlBu_r'):
        if vals is None:
            vals = list(range(2000,2401,100))
        norm = mpl.colors.Normalize(
                vmin=np.min(vals),
                vmax=np.max(vals))
        cm = plt.cm.get_cmap(cmap)
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, ticks=list(set(vals)), location=orientation, pad=pad)
        #cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, ticks=list(set(vals)), orientation=orientation, pad=pad)

        return cm, sm, cb

    def load_data(self, log_fname):
        logs = np.load(log_fname, allow_pickle=True)['arr_0'][()]
        data = []
        for (T, t), sim_data in logs.items():
            sim = SimulationData(sim_data, T, t)
            data.append(sim)

        self.T_range = range(2000, 2401, 100)

        return data



    @staticmethod
    def get_color_T(T, T_min=2000, T_max=2400, cmap='RdYlBu_r'):
        return lammps_logfile.get_color_value(T, T_min, T_max, cmap=cmap)

    @staticmethod
    def get_color_E(E, E_min=0, E_max=10*5.666*0.001, cmap='magma'):
        return lammps_logfile.get_color_value(E, E_min, E_max, cmap=cmap)

    def test_peaks(self):

        N = 51
        T = 2200
        fig, ax = plt.subplots()
        for s in self.data:
            if s.T == T and s.final_time == 100:
                # continue
                print(s.T, s.final_time, s.erate)
                s.smooth_pzz(N)
                s.find_peaks()
                s.find_traughs()
                p = s.peaks[0]
                q = s.traughs[0]
                col = 0.85 / s.final_time
                col = self.get_color_E(col)
                ax.plot(s.strain, s.pzz_raw, lw = 0.1)
                ax.plot(s.strain, s.pzz, color=col)
                ax.plot(s.strain[p], s.pzz[p], "x", color=col, markersize = 10)
                ax.plot(s.strain[q], s.pzz[q], "x", color=col, markersize = 10)
                d = (p + q) // 2
                ax.plot(s.strain[d], s.pzz[d], "x", color=col, markersize = 10)


    def get_bulk_stress_strain(self):

        fname = "/home/Alexander/compsci/thesis/bulk/redo_T2200_25ns/log.lammps"
        log = lammps_logfile.File(fname)

        stress = log.get("Pzz", run_num = 1)
        lz = log.get("Lz", run_num = 1)
        strain = (lz[0] - lz)/lz[0]
        
        return stress*1e-4, strain


    def plot0(self):
        """
        Plots stress/strain curves for:
            * temps 2200 - 2400 K
            * strain rates 1.5 3.75 and 6 1/us
            * bulk vs pore for 2200K
        """

        fig = plt.figure(figsize=(4.7747, 3.9509))
        gs = GridSpec(3, 1)
        ax1 = fig.add_subplot(gs[0,:]) # temps
        ax2 = fig.add_subplot(gs[1,:]) # erate
        ax3 = fig.add_subplot(gs[2,:]) # bulk vs pore

        ax1.plot([0], [0], color="white", label="Temperature", lw = 0)
        ax2.plot([0], [0], color="white", label="Strain rate", lw = 0)
        #ax2.plot([0], [0], color="white", label="Material", lw = 0)
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xlabel("Strain")
        ax1.set_ylabel("Stress [GPa]")
        ax2.set_ylabel("Stress [GPa]")
        ax3.set_ylabel("Stress [GPa]")
        plt.tight_layout()
        gs.update(hspace=0.05)
        for sim in sorted(self.data):
            # N = int(sim.final_time // 8)
            # if N % 2 == 0:
                # N += 1
            N = 7
            sim.smooth_pzz(N = N)
            if sim.final_time == 100:
                ax1.plot(sim.strain, sim.pzz, color = self.get_color_T(sim.T), label=f" {sim.T}" + " K")

        strain_sims = [s for s in self.data if s.T == 2200]
        idx =  np.argsort([s.erate for s in strain_sims])


        for i in idx:
            sim = strain_sims[i]
            # if sim.final_time != 100:
            #     sim.smooth_pzz(N=11)
            erate_col = 0.85 / sim.final_time
            ax2.plot(sim.strain, sim.pzz, color = self.get_color_E(erate_col), label=f" {sim.erate:.2}" + r" ns$^{-1}$")

        for sim in self.data:
            if sim.final_time == 25 and sim.T == 2200:
                ax3.plot(sim.strain, sim.pzz, label="faceted pore", color="tab:blue")
                break

        #bulk 
        b_stress, b_strain = self.get_bulk_stress_strain()
        N = 7
        bulk_smooth = scipy.signal.savgol_filter(b_stress, polyorder=2, window_length=N)
        ax3.plot(b_strain, bulk_smooth, label="bulk material", color="tab:green")

        """
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        """

        #ax1.set_xticklabels([])
        #ax2.set_xlabel("Time [ns]")
        #ax1.set_ylabel("Stress [GPa]")
        #ax2.set_ylabel("Stress [GPa]")
        #plt.tight_layout()
        #ax1.legend(bbox_to_anchor=(1, 0, 0.375, 1), mode='expand')
        #ax2.legend(bbox_to_anchor=(1, 0, 0.375, 1), mode='expand')
        for ax in (ax1, ax2, ax3):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
            ax.legend(bbox_to_anchor=(1, 0, 0.45, 1), mode='expand')
            ax.set_yticks([0, 5, 10])

        ax3.set_yticks([0, 5, 10, 15])

        coords = [(0.05, 0.85)]*3
        #coords = [(0.9, 0.96), (0.9, 0.9), (0.9, 0.9)]
        # x = 0.9
        # y = 0.9
        i = 0
        for ax, label in zip((ax1, ax2, ax3),('a)', 'b)', 'c)')):
            ax.text(*coords[i], label, transform=ax.transAxes)
            i += 1

        plt.savefig("figs/with_bulk_simple_stress_strain_temp_erate.pdf")



    def plot1(self):
        """
        Plots stress/strain curves for temps 2200 - 2400 K and strain rates 1.5 3.75 and 6 1/us
        """

        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,:])

        ax1.plot([0], [0], color="white", label="Temperature", lw = 0)
        ax2.plot([0], [0], color="white", label="Strain rate", lw = 0)
        ax1.set_xticklabels([])
        ax2.set_xlabel("Strain")
        ax1.set_ylabel("Stress [GPa]")
        ax2.set_ylabel("Stress [GPa]")
        plt.tight_layout()
        gs.update(hspace=0.05)
        for sim in sorted(self.data):
            # N = int(sim.final_time // 8)
            # if N % 2 == 0:
                # N += 1
            N = 7
            sim.smooth_pzz(N = N)
            if sim.final_time == 100:
                ax1.plot(sim.strain, sim.pzz, color = self.get_color_T(sim.T), label=f" {sim.T}" + " K")

        strain_sims = [s for s in self.data if s.T == 2200]
        idx =  np.argsort([s.erate for s in strain_sims])

        for i in idx:
            sim = strain_sims[i]
            # if sim.final_time != 100:
            #     sim.smooth_pzz(N=11)
            erate_col = 0.85 / sim.final_time
            ax2.plot(sim.strain, sim.pzz, color = self.get_color_E(erate_col), label=f" {sim.erate:.2}" + r" ns$^{-1}$")

        """
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        """

        #ax1.set_xticklabels([])
        #ax2.set_xlabel("Time [ns]")
        #ax1.set_ylabel("Stress [GPa]")
        #ax2.set_ylabel("Stress [GPa]")
        #plt.tight_layout()
        #ax1.legend(bbox_to_anchor=(1, 0, 0.375, 1), mode='expand')
        #ax2.legend(bbox_to_anchor=(1, 0, 0.375, 1), mode='expand')
        for ax in (ax1, ax2):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
            ax.legend(bbox_to_anchor=(1, 0, 0.45, 1), mode='expand')
            ax.set_yticks([0, 5, 10])


        # plt.savefig("figs/simple_stress_strain_temp_erate.pdf")

    def plot2(self, n_peaks = 3):
        """
        plots yield stress at the consecutive peaks
        """

        for s in self.data:
            s.smooth_pzz(21)
            s.find_peaks()

        # fig1, (ax11, ax22) = plt.subplots(ncols=2, sharey=True)
        fig, ax1 = plt.subplots()
        xy = [(0.075, 8.44), (0.086, 7.9), (0.1052, 7.14)]
        for i in range(n_peaks):
            yield_strain = []
            yield_stress = []
            for s in self.data:
                try:
                    p = s.peaks[i]
                    #ax.scatter(i+1, s.pzz[p], s=20, color = self.get_color_T(s.T))
                    col_T = self.get_color_T(s.T)
                    #col_E = self.get_color_E(s.erate)
                    ax1.scatter(s.strain[p], s.pzz[p], marker = "o", s=20, color=col_T)
                    #ax2.scatter(s.strain[p], s.pzz[p], marker = "o", s=20, color=col_E)
                    yield_strain.append(s.strain[p])
                    yield_stress.append(s.pzz[p])
                except:
                    print(f"No {i}. peak in {s}")
            width = max(yield_strain) -  min(yield_strain)
            if i == 2:
                factor = 1.42
            else:
                factor = 1.2

            height = (max(yield_stress) - min(yield_stress))*factor

            #for k in range(len(xytext[i])):
            #    x0 = xy[i][k]
            #    x1 = xytext[i][k]
            #    ax.annotate(f"{i+1}", x0, xytext=x1, arrowprops = dict(facecolor='black'))

            ax1.annotate(f"{i+1}", xy[i])
            #for k in xy[i]:
            #    ax.scatter(*k, s = 20, c = 'k')


            e = Ellipse((np.mean(yield_strain), np.mean(yield_stress)), width, height, fill=False, color="black", alpha = 0.15, angle = -0.3, linestyle='--')
            ax1.add_artist(e)
                #ax.plot(s.strain, s.pzz, color = self.get_color_T(s.T))
                #ax.scatter(s.strain[s.peaks], s.pzz[s.peaks], s=20, marker="x")

        ax1.set_xlim([0.05, 0.15])
        ax1.set_ylim([5.8, 11])
        ax1.set_xlabel("Yield strain")
        ax1.set_ylabel("Yield stress [GPa]")
        plt.tight_layout()
        plt.savefig("figs/yield_stress_strain_peaks.pdf")

    def plot3(self):
        """
        youngs modulus, yield stress and yield strain vs temp
        """

        errors = {T: [] for T in self.T_range}
        vals = {T: [] for T in self.T_range}
        for s in self.data:
            s.compute_youngs_modulus()
            vals[s.T].append(s.youngs_modulus)
            errors[s.T].append(s.youngs_modulus_err)


        youngs = {}
        youngs_err = {}
        n = 4
        for T in self.T_range:
            youngs[T] = np.mean(vals[T])
            youngs_err[T] = np.std(errors[T])/np.sqrt(len(errors[T]))
            print(f"{T=}: E = {youngs[T]:.1f} ({youngs_err[T]:.1f})")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4)
        for s in self.data:
            s.compute_youngs_modulus()
            ax1.scatter(s.T, s.youngs_modulus, s = 20)

        ax1.set_ylabel("$E$ [GPa]")


        #yield stress and strain
        y_stress = {T: [] for T in self.T_range}
        y_strain = {T: [] for T in self.T_range}
        for s in self.data:
            s.smooth_pzz(21)
            y_stress[s.T].append(s.yield_stress)
            y_strain[s.T].append(s.yield_strain)
            if s.T == 2400:
                ax4.plot(s.strain, s.pzz)
                d = (s.peaks[0] + s.traughs[0]) // 2
                ax4.plot(s.strain[d], s.pzz[d], "kx")
                print(s.yield_strain)

        for T in self.T_range:
            mean_y_stress = np.mean(y_stress[T])
            std_y_stress = np.std(y_stress[T])

            mean_y_strain = np.mean(y_strain[T])
            std_y_strain = np.std(y_strain[T])

            print("="*5)
            print(f"{T=}, {mean_y_stress=:.1f}({std_y_stress:.1f})")
            print(f"{T=}, {mean_y_strain=:.3f}({std_y_strain:.3f})")
            ax2.scatter(T, mean_y_stress, s=20)
            ax3.scatter(T, mean_y_strain, s=20)


        ax2.set_ylabel("Yield stress [GPa]")
        ax3.set_ylabel("Yield strain")
        ax3.set_xlabel("Temperature [K]")

    def plot4(self):

        """plots change in pore volume vs proximity to yield strain v3"""

        num_intervals = 10
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[:,1])
        for s in self.data:
            dE, dV, prox =  s.compute_dV(num_intervals = num_intervals, debug=False)
            ax1.scatter(dE, dV, s=10, c = self.get_color_T(prox, 0, 1, "seismic"), alpha=0.5)#facecolors='none', edgecolors=self.get_color_T(prox, 0, 1))

        ax1.set_ylabel(r"$\Delta V/\Delta \varepsilon$ [nm$^3$]")
        ax1.set_xlabel(r"$\Delta \sigma_{zz}/\Delta \varepsilon$ [GPa]")
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        cm, sm, cb = self.get_cb(fig, ax1, vals=np.arange(0, 1.1, 0.5), cmap="seismic", orientation="top")
        cb.set_label(r"$\varepsilon/\varepsilon^y$")
        plt.tight_layout()


        ax2 = fig.add_subplot(gs[0,0])
        ax3 = fig.add_subplot(gs[1,0])
        for s in self.data:
            if s.T == 2300 and s.final_time == 100:
                p = s.peaks[0]
                i = p + 1000
                #i = s.find_nearest_sorted(s.strain, 0.08)
                s.smooth_vol(N=51)
                tmp_strain = s.strain[:i]/s.strain[p]
                tmp_pzz = s.smooth(s.pzz_raw[:i], N=11)
                ax2.plot(tmp_strain, s.vol[:i], color=self.get_color_T(2390), linewidth=0.4)
                ax3.plot(tmp_strain, tmp_pzz, color=self.get_color_T(2390), linewidth=0.4)


        ax2.set_ylabel(r"$V$ [nm$^3$]")
        ax3.set_xlabel(r"$\varepsilon/\varepsilon^y$")
        ax3.set_ylabel(r"$\sigma_{zz}$ [GPa]")
        #ax2.set_xticklabel([])
        ax2.xaxis.set_ticklabels([])

        plt.tight_layout()
        gs.update(hspace=0.05)
        fig.savefig("figs/strain_volume3.pdf")

    def plot5(self):
        """potential energy during deformation"""

        #fig, (ax1, ax2) = plt.subplots(nrows=2)


        fig, ax1= plt.subplots(nrows=1)
        s = self.data[0]
        #strain_eng = scipy.integrate.cumtrapz(s.pzz, s.strain, initial=0)
        s.smooth_pzz()
        poteng = s.smooth(s.poteng, N=51)
        i = s.find_nearest_sorted(s.t, 18)
        ax1.plot(s.t[:i], poteng[:i]/n_atoms, linewidth=0.4, color=self.get_color_T(2390))
        #ax2.plot(s.t[:i], strain_eng[:i])
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Potential energy [eV]")
        plt.tight_layout()


        arrowprops = dict(width=1, headwidth=5, headlength=5, shrink=1, facecolor='black')
        #arrowprops = dict(arrowstyle="fancy")

        ax1.text(3.1, -5.95, "Compression")
        ax1.text(13.2, -5.928, "Yield")


        ax1.annotate(" ", (5, -5.955), xytext=(5, -5.950), arrowprops = arrowprops)
        ax1.annotate(" ", (16.4, -5.926), xytext=(15, -5.929), arrowprops = arrowprops)

        fig.savefig("figs/poteng.pdf")

    def collapse_stress_strain(self, final_times):
        """
        fits to tau = t*s_rate*exp(E/kT)
        """

        yvals = []
        xvals = []
        for i, s in enumerate(self.data):
            if s.final_time not in final_times:
                continue
            s.smooth_pzz()
            s.find_peaks()
            xvals.append(1/(s.T*k_b))
            # yvals.append(np.log(1/(s.erate*s.t[s.peaks[0]])))
            yvals.append(np.log(1/(s.t[s.peaks[0]])))


        # fig, ax = plt.subplots()
        # ax.plot(xvals, yvals, "ro", alpha=0.5, markersize=10)
        # plt.show()

        fit = poly_fitter(xvals, yvals)
        E_a = -fit["slope"]
        tau0 = np.exp(-fit["const"])

        # plt.scatter(xvals, yvals)
        # for i, s in enumerate(self.data):
        #     if s.final_time not in final_times:
        #         continue

            # plt.plot(s.t*s.erate*np.exp(-E_a/(s.T*k_b)), s.smooth(s.pzz_raw, 11)/s.T**(-2))

        # print(f"{E_a=:.3f}, {tau0=:.3f}")
        return E_a, tau0, fit["error"]

    def plot6(self):
        """
        datacollapse of stress/strain curves
        skipping this for now. model is rubbish. GPa != K^2....
        """

        final_times = [100]

        fig, ax1 = plt.subplots()
        E_a = self.collapse_stress_strain(final_times)
        for s in self.data:
            if s.final_time not in final_times:
                continue
            s.compute_youngs_modulus()
            col = self.get_color_T(s.T)
            pzz = s.smooth(s.pzz_raw, N=21)
            ax1.plot(s.t*s.erate*np.exp(-E_a/(k_b*s.T)), pzz*s.youngs_modulus**(-1), color=col)
            #ax1.plot(s.tau, s.pzz)

        ax1.set_xlabel(r"$t\dot{\varepsilon}\exp\left(-E_a/k_bT\right)$")
        ax1.set_ylabel(r"$\sigma_{zz}T^2$")


        plt.tight_layout()

    def plot7(self):
        """
        data-collapse of the curves

        Single strain rate: E_a=0.26(0.01), tau0=4.857


        """

        # fig, ax1 = plt.subplots(figsize=(4.7747, 2.9509))
        # fig = plt.figure(figsize=(4.7747, 2.9509))
        # gs = GridSpec(2, 1)
        # ax1 = fig.add_subplot(gs[0,:])
        # ax2 = fig.add_subplot(gs[1,:])
        # ax1.set_ylabel(r"$\sigma_{zz}/T^n$ [arb.]")
        # ax2.set_ylabel(r"$\sigma_{zz}/T^n$ [arb.]")
        # ax2.set_xlabel(r"$\varepsilon\exp\left( -E_a / k_bT \right)$")
        # plt.tight_layout()
        # ax1.set_xticklabels([])
        # gs.update(hspace=0.05)

        fig, ax1 = plt.subplots()
        ax1.set_ylabel(r"$\sigma_{zz}/T^n$ [arb.]")
        ax1.set_xlabel(r"$\varepsilon\exp\left( -E_a / k_bT \right)$")



        # single strain rate
        t_final = [40]
        E_a, tau0, err = self.collapse_stress_strain(t_final)
        for s in self.data:
            if s.final_time not in t_final:
                continue
            x = s.strain*np.exp(-E_a/(s.T*k_b))
            y = s.smooth(s.pzz_raw, 11)/s.T**(-2)
            ax1.plot(x, y, color=self.get_color_T(s.T), linewidth=0.5)

        print(f"Single strain rate: E_a={E_a:.3f}({err:.3f}), tau0={tau0:.3f}")
        plt.tight_layout()
        plt.savefig("figs/stress_strain_datacollapse_single_erate.pdf")

        """
        #all data
        t_final = [18.5, 25, 40, 100]
        E_a, tau0, err = self.collapse_stress_strain(t_final)
        for s in self.data:
            if s.final_time not in t_final:
                continue
            x = s.strain*np.exp(-E_a/(s.T*k_b))
            y = s.smooth(s.pzz_raw, 11)/s.T**(-2)
            ax2.plot(x, y*1e-7, color=self.get_color_T(s.T), linewidth=0.4)
        ax1.set_xlim([-0.001, 0.05])
        ax2.set_xlim([-0.001, 0.05])
        print(f"All data: E_a={E_a:.3f}({err:.3f}), tau0={tau0:.3f}")


        plt.savefig("figs/stress_strain_datacollapse.pdf")

        plt.tight_layout()
        """

    def plot8(self):
        """
        plot youngs modulus and yield strength vs temp
        """


        temps = list(range(2000, 2401, 100))
        E = [170.7, 161.1, 152.8, 146.3, 139.8]
        dE = [0.5, 0.5, 0.5, 0.6, 0.6]

        Y = [10.4, 9.7, 9.0, 8.2, 7.5]
        dY = np.array([1, 1, 2, 2, 1])*0.1

        fig = plt.figure()
        gs = GridSpec(2, 1)

        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax.errorbar(temps, E, dE, color = "tab:blue", capsize=1)
        ax2.errorbar(temps, Y, dY, color = "tab:red", capsize=1)


        ax.set_ylabel(r"$E$ [GPa]")
        ax2.set_ylabel(r"$\sigma^y$ [GPa]")
        ax2.set_xlabel(r"$T$ [K]")
        ax.set_xticklabels([])
        plt.tight_layout()
        
        ax.set_xticks(temps) 
        ax2.set_xticks(temps) 
        
        gs.update(hspace=0.1)

        plt.savefig("figs/youngs_mod_yield_stress_vs_temps.pdf")

        


class Creep:
    def __init__(self, log_fname:str):
        self.data = self.load_data(log_fname)

    def load_data(self, log_fname):
        logs = np.load(log_fname, allow_pickle=True)['arr_0'][()]
        data = []
        for (T, t), sim_data in logs.items():
            sim = SimulationData(sim_data, T, t)
            data.append(sim)

        self.T_range = range(2000, 2401, 100)

        return data
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def plot(self):

        for sim in self.data:
            if sim.T == 2000 and sim.final_time == 25:
            #if sim.T == 2100 and sim.final_time == 25:
                s = sim
                break
        else:
            print("no sims found..")
            return

        pzz = s.smooth(s.pzz, 3)
        yield_time = 13.58

        idx = self.find_nearest(s.t, yield_time*0.6) 

        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(4.7747, 3.9509))


         
        
        x1 = s.t[:idx]
        y1 = pzz[:idx]

        idx2 = self.find_nearest(s.t, yield_time)
        idx3 = self.find_nearest(s.t, yield_time*1.1)

        x2 = s.t[idx:idx2]
        y2 = pzz[idx:idx2]

        #y3 = pzz[idx2-1:idx3]
        pzz2 = s.smooth(pzz, 31)
        n = 10
        x3 = s.t[idx2-n:idx3]+0.1
        y3 = pzz2[idx2-n:idx3]

        ax.plot(x1, y1, color="tab:blue", label="initial compression")
        ax.plot(x2, y2, color="tab:green", label="restart files generated")
        ax.plot(x3, y3, "k--", label="post yield")
        #ax.plot([s.t[idx]]*2, [0, 11], "k--")
        #ax.plot([s.t[idx2]]*2, [0, 11], "k--")
        ax.legend()
        ax.set_ylabel(r"$\sigma_{zz}$ [GPa]")
        ax.set_xlabel(r"$t$ [ns]")
        #plt.tight_layout()
        #plt.savefig("creep_init.pdf")


        path = "../creep/data/7_log.lammps"
        #path = "../creep/data/5_run5/log.lammps"
        log = lammps_logfile.File(path)
        all_pzz = []
        all_t = []
        for q in range(1, log.get_num_partial_logs()-0):
            t = log.get("Time", run_num=q)*1e-3
            pzz_ = log.get("Pzz", run_num=q)*1e-4
            #pzz = s.smooth(pzz, 101)*1e-4
            all_pzz.extend(pzz_)
            all_t.extend(t)
            #ax2.plot(t, pzz, "r")


        p = 50
        all_pzz = s.smooth(np.asarray(all_pzz)[::p], 3)
        #all_pzz = all_pzz[::p]
        all_t = all_t[::p]
    
        
        m = 100
        xx = np.array(x2) - x2[-1]
        xx = list(xx)[m:]
        yy = list(y2)[:-m]

        

        ax2.plot(all_t, all_pzz, label="creep run") 

        tt = 10.8
        off = 0.25
        yt = 5.527
        ax2.plot([0, yt], [tt, tt], "k")
        ax2.plot([0,0], [tt-off, tt+off], "k")
        ax2.plot([yt, yt], [tt-off, tt+off], "k")


        c = (yt/2 - 0, tt + off)
        label = r"$\tau$"
        print(c)
        ax2.text(c[0], c[1], label)#, transform=ax2.transAxes)

        ax2.plot(xx, yy, "grey", label="history")

        ax2.set_ylabel(r"$\sigma_{zz}$ [GPa]")
        ax2.set_xlabel(r"$t$ [ns]")
        ax2.legend()
        ax2.set_ylim([6.3, 12.25])
        plt.tight_layout()
        
        plt.savefig("creep_init.pdf")

        


def main():
    rdir1= "/home/alexander/compsci/thesis/dev/SiC_inverted_crystals/deform/proper_init_temp/completed"
    # rdir2 = "/home/alexander/compsci/thesis/dev/SiC_inverted_crystals/deform/with_npt/completed/"
    # rdir3 = "/home/alexander/compsci/thesis/dev/SiC_inverted_crystals/deform/POTENG"

    out_name1 = "data/logs_proper.npz"
    # out_name2 = "data/logs_npt.npz"
    # out_name3 = "data/logs_poteng.npz"
    # WriteData(rdir1).write(out_name1)
    #WriteData(rdir2).write(out_name2)
    #writer = WriteData(rdir3)
    #writer.set_keywords(["Time", "Pzz", "Lz", "PotEng"])
    #writer.write(out_name3, run_nums = [0, 1])

    plotter = Plotter(out_name1)
    times = [18.2, 25, 40, 100]
    times = [40]
    # plotter.collapse_stress_strain(times)
    # plotter.test_peaks()
    # plotter.plot1()
   # plotter.plot2()
    # plotter.plot3()
    # plotter.plot4()
    #plotter.plot4()
    # plotter.plot6()
    #plotter.plot7()
    #plotter.plot0()
    plotter.plot8()

    plt.show()




if __name__ == '__main__':
    #main()
    
    plotter = Creep("data/logs_proper.npz")
    plotter.plot()
    plt.show()










    #
