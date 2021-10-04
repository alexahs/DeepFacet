import numpy as np
from scipy.optimize import curve_fit


def charge_beta(t, p0, tau, beta):
    return p0*(1 - np.exp(-(t/tau)**beta))

def charge_beta_fixed(t, p0, tau):
    beta = 0.5
    return p0*(1 - np.exp(-(t/tau)**beta))

def discharge_beta_fixed(t, A, tau, U0):
    beta = 0.5
    return A*np.exp(-(t/tau)**beta) + U0

def discharge_beta_A_fixed(t, tau, U0):
    beta = 0.5
    A = 5.842e-03 #20 runs
    return A*np.exp(-(t/tau)**beta) + U0

def discharge(t, A, tau, beta, U0):
    return A*np.exp(-(t/tau)**beta) + U0


def curve_fitter(func, xData, yData, p0=None, bounds=(-np.inf, np.inf)):
    params, cov = curve_fit(func, xData, yData, method='lm', p0=p0, bounds=bounds)
    params = tuple(params)
    x = np.linspace(min(xData), max(xData), 500)
    d = {
        "params": params,
        "cov": cov,
        "fit_x": x,
        "fit_y": func(x, *params)
    }
    return d

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

def girdspec_ex():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)

    # demo 3 : gridspec with subplotpars set.

    f = plt.figure()

    plt.suptitle("GridSpec w/ different subplotpars")

    gs1 = GridSpec(3, 3)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:-1, :])
    ax2 = plt.subplot(gs1[-1, :-1])
    ax3 = plt.subplot(gs1[-1, -1])

    gs2 = GridSpec(3, 3)
    gs2.update(left=0.55, right=0.98, hspace=0.05)
    ax4 = plt.subplot(gs2[:, :-1])
    ax5 = plt.subplot(gs2[:-1, -1])
    ax6 = plt.subplot(gs2[-1, -1])

    make_ticklabels_invisible(plt.gcf())

    plt.show()
