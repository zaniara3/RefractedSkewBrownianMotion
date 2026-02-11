import numpy as np
from sbm_truncnorm.truncnorm import TruncatedNormal


def sample_mixture_truncnorm(params_full, n, seed=0):
    alpha_neq, loc_neg, scale_neg, alpha_pos, loc_pos, scale_pos = params_full
    rng = np.random.default_rng(seed)

    u = rng.random(n)
    mask_neg = u <= alpha_neq

    u_neg = u[mask_neg] / alpha_neq
    u_pos = (u[~mask_neg] - alpha_neq) / alpha_pos

    sample = np.zeros(n, dtype=float)

    tn_neg = TruncatedNormal([loc_neg, scale_neg], -np.inf, 0.0)
    tn_pos = TruncatedNormal([loc_pos, scale_pos], 0.0, np.inf)

    sample[mask_neg] = tn_neg.rand_truncated_normal(u_neg)
    sample[~mask_neg] = tn_pos.rand_truncated_normal(u_pos)
    return sample
