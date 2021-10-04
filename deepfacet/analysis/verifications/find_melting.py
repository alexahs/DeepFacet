import lammps_logfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
plt.style.use('../../../my_ggplot.mplstyle')





def find_melting_Epp_v(path):
    n_temps = 19
    n_atoms = 4096
    log = lammps_logfile.File(path)
    volume = []
    energy = []
    # temp = []
    press = []
    run_idx = [(i*2 + 1) for i in range(n_temps)]
    temps = np.arange(300, 3901, 200)
    for k, i in enumerate(run_idx):
        volumes = log.get("Volume", run_num=i)
        energies = log.get("TotEng", run_num=i)
        # temps = log.get("Temp", run_num=i)
        pressure = log.get("Press", run_num=i)

        volume.append(np.mean(volumes))
        energy.append(np.mean(energies))
        # temp.append(np.mean(temps))
        press.append(np.mean(pressure))



    #vol at zero temperature
    V0 = 42382.777

    volume = np.array(volume)
    energy = np.array(energy)
    press = np.array(press)

    diffs_E = np.diff(energy/n_atoms)
    diffs_V = np.diff(volume/V0)


    idx_E = np.argmax(np.diff(energy/n_atoms))
    idx_V = np.argmax(np.diff(volume/V0))

    melting = (temps[idx_E] + temps[idx_E+1])/2
    error = abs(temps[idx_E] - temps[idx_E+1])/2

    print(f"T_melt = {melting:.1f} ({error:.1f})")
    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(temps, energy/n_atoms, 'r--o', markersize=2)
    ax1.set_ylabel(r"$E_{pp}$ [eV]")

    ax2.plot(temps, volume/V0, 'b-->', markersize=2)
    ax2.set_ylabel(r"$V/V_0$")
    ax2.set_xlabel(r"$T$ [K]")

    ax1.plot([melting, melting], [-4, -7], 'k--', label=rf'Melting temperature: {melting:.0f}({error:.0f}) K')
    ax2.plot([melting, melting], [0.9, 1.8], 'k--')
    ax1.set_ylim([-6.4, -4.2])
    ax2.set_ylim([0.95, 1.7])
    ax1.legend()
    ax1.xaxis.set_ticklabels([])
    plt.tight_layout()
    gs.update(hspace=0.05)
    plt.savefig("SiC_melting_temperature.pdf")
    plt.show()



def main():
    path = "log.lammps"
    find_melting_Epp_v(path)


if __name__ == '__main__':
    main()



#
