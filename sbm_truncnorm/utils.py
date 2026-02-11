import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ecdf

from sbm_truncnorm.truncnorm import normal_truncated_pos
from sbm_truncnorm.truncnorm import normal_truncated_neq


def objective_function_negative_quantile(params , x, sbm_values):
    alpha_neq, loc_neg, scale_neg = params
    err_negative = normal_truncated_neq(alpha_neq, loc_neg, scale_neg, x) - sbm_values
    obj_neg = np.quantile(np.abs((err_negative)),0.99)
    return obj_neg

def objective_function_positive_quantile(params , x, sbm_values, alpha_neg ):
    loc_pos, scale_pos = params
    err_positive = alpha_neg + normal_truncated_pos(1-alpha_neg, loc_pos, scale_pos, x) - sbm_values
    obj_pos = np.quantile(np.abs((err_positive)),0.99)
    return obj_pos


def plot_hist_vs_pdf(sample, x, sbm_pdf, outpath=None, xlim=(-10, 10), bins=200):
    plt.figure()
    plt.hist(sample[(sample >= xlim[0]) & (sample <= xlim[1])], bins=bins, density=True)
    plt.plot(x[x < 0], sbm_pdf[x < 0], color="red")
    plt.plot(x[x > 0], sbm_pdf[x > 0], color="red")
    plt.xlim(list(xlim))
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Histogram of Simulated Data and PDF of SBM")
    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()

def plot_cdf_theory_vs_emp(sample, sbm_cdf_sample_sorted, outpath=None):
    sample_sorted = np.sort(sample)

    plt.figure()
    plt.plot(sample_sorted, sbm_cdf_sample_sorted, label="Theoretical CDF", color="red")
    e = ecdf(sample)
    plt.plot(e.cdf.quantiles, e.cdf.probabilities, label="Empirical CDF", linestyle="dashed")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Theoretical CDF vs Empirical CDF")
    plt.legend()
    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()