import numpy as np
from scipy.optimize import minimize
from .utils import objective_function_negative_quantile, objective_function_positive_quantile

def fit_truncated_normal_to_sbm_cdf(x, cdf_sbm, Mn_sbm, Mp_sbm, t_sbm, beta_sbm, maxiter=100000):
    x = np.asarray(x)
    cdf_sbm = np.asarray(cdf_sbm)

    # Negative side
    params_int_neg = [(1 - beta_sbm) / 1, np.sign(Mn_sbm) * min(abs(Mn_sbm), abs(Mp_sbm)), t_sbm / 3.5]
    bnds_neg = ((0, 1), (None, None), (0, None))

    res_neg = minimize(
        objective_function_negative_quantile,
        params_int_neg,
        args=(x[x < 0], cdf_sbm[x < 0]),
        method="SLSQP",
        bounds=bnds_neg,
        options={"maxiter": maxiter},
    )

    # Positive side
    params_int_pos = [np.sign(Mp_sbm) * min(abs(Mn_sbm), abs(Mp_sbm)), t_sbm]
    bnds_pos = ((None, None), (0, None))

    res_pos = minimize(
        objective_function_positive_quantile,
        params_int_pos,
        args=(x[x > 0], cdf_sbm[x > 0], res_neg.x[0]),
        method="SLSQP",
        bounds=bnds_pos,
        options={"maxiter": maxiter},
    )

    alpha_neg, loc_neg, scale_neg = res_neg.x
    loc_pos, scale_pos = res_pos.x
    alpha_pos = 1.0 - alpha_neg

    params_full = np.array([alpha_neg, loc_neg, scale_neg, alpha_pos, loc_pos, scale_pos], dtype=float)
    return params_full, res_neg, res_pos
