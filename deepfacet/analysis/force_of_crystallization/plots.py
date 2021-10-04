import numpy as np
import lammps_logfile
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from pathlib import Path
from fitting import *
from scipy.signal import savgol_filter
# plt.style.use("ggplot")
plt.style.use("/home/alexander/compsci/thesis/my_ggplot.mplstyle")
# plt.style.use("/home/alexander/compsci/thesis/stylesheet_2.mplstyle")
k_b = 8.617333262145e-5 #eV/K


class PotEng:
    def __init__(self,
                 temps = list(range(2100, 2301, 50)),
                 smooth_window = 5):
        self.temps = temps
        self.smooth_window = smooth_window

    @staticmethod
    def params(fit_type:str):
        if fit_type == "all":
            # A, tau, beta, U0
            return np.array([
            [0.005927624028263969, 2.4945959228855954, 0.499347343611513, -5.976113088291125],
            [0.007910522023131, 1.0082469751128018, 0.3554243034002912, -5.965883756129715],
            [0.00723176165428397, 1.0258759120382066, 0.39576038588067963, -5.955026370866495],
            [0.0071127728025409745, 1.0471491298976345, 0.3928430471364952, -5.944033628987735],
            [0.008554011908079138, 0.6180181984402442, 0.31909147438984714, -5.932899246211169]
            ])

        elif fit_type == "beta":
            # A, tau, U0
            return np.array([
            [0.00592178553418905, 2.497730896884454, -5.976112033737522],
            [0.0059261432349301995, 1.7407747647136553, -5.965641579683963],
            [0.005936512706670368, 1.512273720409162, -5.954895779053593],
            [0.005766198874240747, 1.5921825585979268, -5.9439049197787],
            [0.005661775111386124, 1.535513145452294, -5.932636833195592]
            ])
        elif fit_type == "beta_A":
            # tau, U0
            return np.array([
            [2.579719166774341, -5.976121915590249],
            [1.7966609755767333, -5.965649556629903],
            [1.5631098730389594, -5.954903028144746],
            [1.549774866553356, -5.943899006921214],
            [1.4373571172942625, -5.932622518196809]
            ])

        elif fit_type == "all_partial":
            #fitted to first 10 ns, A, tau, beta, U0
            return np.array([
            [0.005434549374617874, 2.2188991953350437, 0.5558858431604639, -5.975814063885717],
            [0.005550001428342365, 1.118586983255809, 0.5925831206984375, -5.965092691569415],
            [0.005074103350250985, 1.1561902787619938, 0.6709204537436305, -5.954329328828133],
            [0.006721882120582155, 0.9350843655990108, 0.42525882931488534, -5.9437874397916755],
            [0.005293327023646019, 0.7522202544607463, 0.6421932183987719, -5.931861926016423]
            ])
        else:
            raise NotImplementedError

    def errors(self, fit_type:str):
        if fit_type == "beta_A":
            return np.array([0.004552219021881273,
                             0.00287673593250179,
                             0.0024937768173669238,
                             0.0025655472520972294,
                             0.0024455222669510883])
        else:
            raise NotImplementedError

    def plot1(self):
        """
        fit all params (U0, A, tau, beta) to full and partial data
        """

        plot_inter = 10

        pp = None

        fit_func = discharge
        fig, ax = plt.subplots(figsize = (4.77, 5))
        cm, sm, cb = Misc.get_cb(fig, ax, orientation="horizontal", pad=0.15)

        flag1, flag2 = (True, True)
        for div in (1, 3):
            betas = []
            for i, T in enumerate(self.temps):
                time, P, U, msd = np.load(f"data/T{T}.npy")
                time *= 0.001
                n_fit = len(time) // div

                U_raw = U/12457
                U_smooth = lammps_logfile.running_mean(U_raw, N=self.smooth_window)
                if div == 1:
                    p = self.params("all")[i]
                elif div == 3:
                    p = self.params("all_partial")[i]

                fit_x = np.linspace(0, time[-1], 500)
                fit_y = fit_func(fit_x, *p)
                betas.append(p[2])


                if div == 1:
                    color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
                    ax.plot(time[::plot_inter], U_smooth[::plot_inter], alpha = 1, color = color, label=None, linewidth=0.5)
                    col2 = 'k'
                    if flag1:
                        label = 'fits to full dataset'
                        flag1 = False
                    else:
                        label = None
                else:
                    col2 = 'tab:gray'
                    if flag2:
                        label = 'fits to first 10 ns'
                        flag2 = False
                    else:
                        label = None
                ax.plot(fit_x, fit_y, linestyle='dashed', color=col2, label=label, linewidth=1)

            print(f"{div=}: {np.mean(betas):.2f} ({np.std(betas):.2f})")
        ax.set_ylabel("Potential energy [eV]")
        ax.set_xlabel("Time [ns]")
        plt.tight_layout()
        ax.legend(loc='best')#, ncol=2, bbox_to_anchor=(0, 0.98, 1, 0.2), mode='expand')




        plt.savefig("plots/poteng_fit_all.pdf")
        # plt.show()

    def plot2(self):
        """
        fits to potential energy with tau and U0 as the only free parameters
        shows 3 subplots:
        * data + fit
        * arrhenius plot
        * data collapse
        """
        plot_inter = 10
        # fig, ax = plt.subplots(ncols=2)
        fig = plt.figure(figsize = (4.7747, 4.5))#, constrained_layout=True)
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[:,0])

        for i, T in enumerate(self.temps):
            time, P, U, msd = np.load(f"data/T{T}.npy")
            time *= 0.001

            U_raw = U/12457
            U_smooth = lammps_logfile.running_mean(U_raw, N=self.smooth_window)
            p = self.params("beta_A")[i]
            err = self.errors("beta_A")[i]
            # fit = curve_fitter(discharge_beta_A_fixed, time, U_raw)
            # err = np.sqrt(fit["cov"][0,0])
            # print(f"{err=}")
            # p = fit["params"]

            fit_x = np.linspace(0, time[-1], 500)
            fit_y = discharge_beta_A_fixed(fit_x, *p)

            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            ax1.plot(time[::plot_inter], U_smooth[::plot_inter], alpha = 1, color = color, label=None, linewidth=0.5)
            if i == 0:
                label = f"fit to full dataset"
            else:
                label = None
            ax1.plot(fit_x, fit_y, linestyle='dashed', color='k', label=label, linewidth=0.8)

        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Potential energy [eV]")
        ax1.set_xticks([0, 15, 30])
        # ax1.text(28.4, -5.925, "a)", fontweight="bold", weight="bold")

        xvals = []
        yvals = []
        ax2 = fig.add_subplot(gs[0, 1])
        for i, T in enumerate(self.temps):
            tau = self.params("beta_A")[i,0]
            print(tau)
            x = 1/(k_b*T)
            y = np.log(1/tau)
            xvals.append(x)
            yvals.append(y)
            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            ax2.scatter(x, y, color = color, s=20)
            # ax2.errorbar(x, y, yerr=z, color = color, ms=100, capsize=2)

        ax2.set_xlabel(r"$(k_BT)^{-1}$ [eV$^{-1}$]")
        ax2.set_ylabel(r"ln$(1/\tau)$")

        fit = poly_fitter(xvals, yvals)
        print(f"E_a = {-fit['slope']:.2f}({fit['error']:.2f})")
        ax2.plot(fit["fit_x"], fit["fit_y"], 'k--', label="line fit", linewidth=0.8)
        # ax2.text(5.5, -0.315, "b)")
        ax2.set_xticks([5.1, 5.3, 5.5])

        ax3 = fig.add_subplot(gs[1,1])
        # cm, sm, cb = Misc.get_cb(fig, ax2)


        i = 4
        for T in np.flip(self.temps):
            time, P, U, msd = np.load(f"data/T{T}.npy")
            time *= 0.001
            U /= 12457
            U = lammps_logfile.running_mean(U, N=self.smooth_window*1000)

            tau = self.params("beta_A")[i,0]
            A = self.params("beta")[i,0]
            beta = 0.5
            U0 = self.params("beta_A")[i,1]

            u = (U - U0)/A
            t = (time/tau)#**beta

            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            n = 100
            ax3.plot(t[::n], u[::n], color=color, linewidth=0.5)
            i -= 1

        ax3.set_xlabel(r"$t/\tau$")
        ax3.set_ylabel(r"$(U(t) - U_0)/A$")
        ax3.set_yticks([0, 0.5, 1])
        # ax3.text(20, 1.0, "c)")

        coords = [(0.9, 0.96), (0.9, 0.9), (0.9, 0.9)]
        # x = 0.9
        # y = 0.9
        i = 0
        for ax, label in zip((ax1, ax2, ax3),('a)', 'b)', 'c)')):
            ax.text(*coords[i], label, transform=ax.transAxes)
            i += 1


        plt.tight_layout()
        plt.savefig("plots/poteng_final_fit.pdf")
        plt.show()


class Press:
    def __init__(self,
                 temps = list(range(2100, 2301, 50)),
                 smooth_window = 1000):
        self.temps = temps
        self.smooth_window = smooth_window

    @staticmethod
    def params(fit_type:str):
        if fit_type == "beta_fixed":
            return np.array([[35.99104906596837, 4.166104755297522],
                            [34.11772638816134, 2.4751847708716745],
                            [33.27198108042231, 1.343830207202716],
                            [32.781195382538684, 2.15294587051344],
                            [32.83687052951652, 1.5555110513075343]])
        else:
            raise NotImplementedError

    def plot1(self):
        """
        fits to pressure with tau fixed to 0.5
        shows 3 subplots:
        * data + fit
        * arrhenius plot
        * data collapse
        """
        plot_inter = 10
        # fig, ax = plt.subplots(ncols=2)
        fig = plt.figure(figsize = (4.7747, 4.5))#, constrained_layout=True)
        gs1 = GridSpec(5, 1)
        gs1.update(left=0.10, right=0.48, hspace=0.05)
        # ax1 = fig.add_subplot(gs[:,0])
        p_ax = []

        for i, T in enumerate(self.temps):
            ax = fig.add_subplot(gs1[i,0])
            p_ax.append(ax)
            time, P, U, msd = np.load(f"data/T{T}.npy")
            time *= 0.001
            P *= 0.1

            P_smooth = lammps_logfile.running_mean(P, N=self.smooth_window)
            p = self.params("beta_fixed")[i]

            # fit = curve_fitter(charge_beta_fixed, time, P)
            # p = fit["params"]
            # dP0 = fit["cov"][0,0]
            # dTau = fit["cov"][1,1]
            # print(p)
            # print(f"{dTau=:4f}, {dP0=:.4f}")

            fit_x = np.linspace(0, time[-1], 500)
            fit_y = charge_beta_fixed(fit_x, *p)

            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            ax.plot(time[::plot_inter], P_smooth[::plot_inter], alpha = 1, color = color, label=f"{T} K", linewidth=0.5)
            # if i == 0:
                # label = f"fit to full dataset"
            # label = f"{T} K"
            # else:
                # label = None
            ax.plot(fit_x, fit_y, linestyle='dashed', color='k', label=None, linewidth=0.8)
            ax.set_yticks([0, 25, 50])
            ax.set_ylim(-15, 55)
            if i != 4:
                ax.set_xticklabels([])
            ax.set_xticks([0, 15, 30])

        p_ax[4].set_xlabel("Time [ns]")
        p_ax[2].set_ylabel("Pressure [MPa]")
        p_ax[4].set_xticks([0, 15, 30])
        # return

        xvals = []
        yvals = []
        gs2 = GridSpec(2, 1)
        gs2.update(left=0.65, right=0.98, hspace=0.4)
        ax2 = fig.add_subplot(gs2[0, 0])
        # arrhenius
        for i, T in enumerate(self.temps):
            tau = self.params("beta_fixed")[i,1]
            # print(tau)
            x = 1/(k_b*T)
            y = np.log(1/tau)
            xvals.append(x)
            yvals.append(y)
            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            ax2.scatter(x, y, color = color, s=20)



        ax2.set_xlabel(r"$(k_BT)^{-1}$ [eV$^{-1}$]")
        ax2.set_ylabel(r"ln$(1/\tau)$")

        fit = poly_fitter(xvals, yvals)
        print(f"E_a = {-fit['slope']:.2f}({fit['error']:.2f})")
        ax2.plot(fit["fit_x"], fit["fit_y"], 'k--', label="line fit", linewidth=0.8)
        ax2.set_xticks([5.1, 5.3, 5.5])
        ax3 = fig.add_subplot(gs2[1,0])

        i = 4
        #pressure time avg
        for T in self.temps:
            time, P, U, msd = np.load(f"data/T{T}.npy")
            time *= 0.001
            P = lammps_logfile.running_mean(P, N=int(self.smooth_window*12.5))*0.1

            P0, tau = self.params("beta_fixed")[i]
            # P0 = self.params("beta_fixed")[i,0]

            # p = P/P0
            # t = (time/tau)#**beta

            color = lammps_logfile.get_color_value(T, min(self.temps), max(self.temps), cmap='RdYlBu_r')
            n = 10
            ax3.plot(time[::n], P[::n], color=color, linewidth=0.5)
            i -= 1

        # ax3.set_xlabel(r"$t/\tau$")
        # ax3.set_ylabel(r"$(U(t) - U_0)/A$")
        ax3.set_xlabel("Time [ns]")
        ax3.set_ylabel("Pressure [MPa]")
        ax3.set_yticks([0, 10, 20, 30])
        ax3.set_xticks([0, 15, 30])
        # ax3.text(20, 1.0, "c)")

        coords = [(0.9, 1.1), (0.05, 0.05), (0.05, 0.9)]
        # x = 0.9
        # y = 0.9
        i = 0
        for ax, label in zip((p_ax[0], ax2, ax3),('a)', 'b)', 'c)')):
            ax.text(*coords[i], label, transform=ax.transAxes)
            i += 1


        # plt.tight_layout()
        plt.savefig("plots/pressure_fits.pdf")
        plt.show()


class Misc:
    def __init__(self):
        pass

    @staticmethod
    def err(x, dx):
        mean = np.sum(x*(1/dx)**2)/np.sum((1/dx)**2)
        err = 1/np.sum((1/dx)**2)**0.5
        return (mean, err)

    @staticmethod
    def get_cb(fig, ax, temps=list(range(2100,2301,50)), orientation="vertical", pad=0.1):

        norm = mpl.colors.Normalize(
                vmin=np.min(temps),
                vmax=np.max(temps))
        cm = plt.cm.get_cmap('RdYlBu_r')
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, ticks=list(set(temps)), orientation=orientation, pad=pad)
        cb.set_label("Temperature [K]")

        return cm, sm, cb

    @staticmethod
    def surface_energy(U, area, n=(100, 100, 25)):
        n1, n2, n3 = n

        U0 = np.mean(U[:n1])
        U0_std = np.std(U[:n1])

        U1 = np.mean(U[-n1:])
        U1_std = np.std(U[-n1:])

        A = np.mean(area[-n3:])
        A_std = np.std(area[-n3:])

        surface_eng = (U1 - U0)/A



        err_U = np.sqrt(U0_std**2 + U1_std**2)
        err_A = A_std

        err_gamma = np.sqrt((err_U/(U0 + U1))**2 + (err_A / A)**2)*surface_eng



        print(f"{U0=}")
        print(f"{U1=}")
        print(f"{(U1-U0)=}")
        print(f"{A=}")
        print(f"{surface_eng=:.3f}({err_gamma:.3f})")


    @staticmethod
    def diffusion(t, msd, d=2, n_points=300):

        step = len(msd) // n_points
        print(step)

        D_vals = np.zeros(n_points)
        t_vals = np.zeros(n_points)
        for i in range(n_points):
            beg = i*step
            end = beg + step
            D_vals[i] = np.polyfit(t[beg:end], msd[beg:end], deg=1)[0] / (2*d)
            t_vals[i] = t[beg]

        D_vals *= 1e-7

        return t_vals, D_vals

    def press_poteng_msd_area(self, T, windows=(250, 5)):

        surface_eng_factor = 1.60218e-1 #eV/nm^2 to J/m^2

        t, P, U, msd = np.load(f"data/T{T}.npy")
        t1, area = np.load(f"data/time_surface_area_T{T}.npy")
        t1 *= 0.001
        t *= 0.001
        P *= 0.1
        msd -= msd[0]

        area *= 0.01
        surface_eng = self.surface_energy(U, area, n=(5000, 5000, 25))

        U /= 12457

        t2, D = self.diffusion(t, msd)

        P_smooth = lammps_logfile.running_mean(P, N=windows[0])
        # P_smooth = P
        U_smooth = lammps_logfile.running_mean(U, N=windows[1])
        # msd = savgol_filter(msd, polyorder=2, window_length=10001)
        fig, ax = plt.subplots(nrows = 2, ncols=2, sharex = True, figsize=(4.7747, 3.5))

        ax = ax.flat

        col = []
        for i in range(4):
            col.append(lammps_logfile.get_matlab_color(i))

        lw = 0.4
        n_plot = 100
        ax[0].plot(t[::n_plot], P_smooth[::n_plot], color=col[0], linewidth=lw)
        ax[1].plot(t1, area, color=col[2], linewidth=1)
        ax[2].plot(t[::n_plot], U_smooth[::n_plot], color=col[1], linewidth=lw)
        ax[3].scatter(t2, D, color = col[3], alpha=0.5, s=10)
        # print(f"{np.mean(P[-50000:])}")
        # print(t[-50000])
        # ax[3].plot(t[::n_plot], msd[::n_plot], color = col[2], linewidth=1)
        n_area = 50
        # ax[3].plot(t1[:-n_area], area[:-n_area], color=col[3], linewidth=1)
        # ax[3].plot(t1[-n_area:], area[-n_area:], color="black", linewidth=1)
        ax[0].set_ylabel("Pressure [MPa]")
        ax[1].set_ylabel(f"Surface area [nm$^2$]")
        ax[2].set_ylabel("Potential\nenergy [eV]")
        ax[3].set_ylabel("Mass\n"+r"diffusivity [cm$^2/$s]")
        # ax[3].set_ylabel("Mean squared\n"+r" displacement [Å$^2$]")
        # ax[3].set_ylabel("Mass\n"+r"diffusivity [Å$^2/$ns]")
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")

        # ax[2].set_ylabel(r"$\langle r^2 \rangle$ [Å$^2$]")
        ax[2].set_xlabel("Time [ns]")
        ax[2].set_xticks(list(range(0, 31, 10)))
        # ax[2].set_yticks([0, 5, 10, 15])

        ax[3].set_xticks(list(range(0, 31, 10)))
        ax[3].set_xlabel("Time [ns]")
        ax[3].yaxis.tick_right()
        ax[3].yaxis.set_label_position("right")

        # fig = plt.figure()
        # for i, label in enumerate(('A', 'B', 'C', 'D')):
        #     ax = fig.add_subplot(2, 2, i+1)
        #     ax.text(0.05, 0.95, label, transform=ax.transAxes,
        #             fontsize=16, fontweight='bold', va='top')
        #
        # plt.show()

        coords = [(0.825, 0.1), (0.1, 0.1), (0.825, 0.9), (0.1, 0.9)]

        for i, label in enumerate(('a)', 'b)', 'c)', 'd)')):
            ax[i].text(*coords[i], label, transform=ax[i].transAxes)

        fig.align_ylabels(ax)
        plt.tight_layout()
        # plt.savefig(f"plots/press_poteng_msd_area_T{T}.pdf")
        # plt.savefig(f"plots/press_poteng_msd_area_T{T}.pgf")
        # plt.show()





if __name__ == '__main__':
    # p = PotEng().plot1()
    # p = PotEng().plot2()
    p = Misc().press_poteng_msd_area(2200)
    #Press().plot1()
    #plt.show()
