import numpy as np
import matplotlib.pyplot as plt
import lammps_logfile

plt.style.use('../../../my_ggplot.mplstyle')

def poly_fitter(x, y):
    p, cov = np.polyfit(x, y, deg=1, cov=True)
    fit_x = [np.min(x), np.max(x)]
    fit_y = np.polyval(p, fit_x)
    err = np.sqrt(np.diag(cov))[0]
    dict = {
        "slope": p[0],
        "const": p[1],
        "error": err,
        "fit_x": fit_x,
        "fit_y": fit_y
    }
    return dict


def youngs_mod():
    log = lammps_logfile.File("log.lammps")
    run_num = 1


    pzz = log.get("Pzz", run_num = run_num)*1e-4
    lz = log.get("Lz", run_num = run_num)
    strain = abs(lz - lz[0])/lz[0]
    n = len(strain) // 3
    p, cov = np.polyfit(strain[:n], pzz[:n], deg=1, cov=True)
    # err = np.sqrt(np.sum(np.diag(cov)))
    err = np.sqrt(np.diag(cov))[0]
    youngs_mod = p[0]
    print(f"{youngs_mod=:.2f} ({err:.2f})")
    plt.title("Youngs modulus")
    plt.xlabel("strain")
    plt.ylabel("stress")


    plt.plot(strain, pzz)
    plt.tight_layout()
    plt.show()



def elasticity_tensor():
    log = lammps_logfile.File("log.lammps")
    run_num = 1

    U = log.get("PotEng", run_num = run_num)
    lz = log.get("Lz", run_num = run_num)
    strain = abs(lz - lz[0])/lz[0]
    vol = log.get("Volume", run_num = run_num)

    psi = U/vol

    dPsi = np.gradient(psi, strain)
    d2Psi = np.gradient(dPsi, strain)
    plt.plot(psi, label="psi")
    plt.plot(dPsi, label="dPsi")
    plt.plot(d2Psi, label="d2Psi")
    plt.legend()
    plt.show()



def main():
    # elasticity_tensor()
    youngs_mod()



if __name__ == '__main__':
    main()
