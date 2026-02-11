import numpy as np
from scipy.stats import norm


class TruncatedNormal():
    def __init__(self,params, a , b):
        self.location = params[0]
        self.scale = params[1]
        self.a = a
        self.b = b

    def normal_pdf(self,x):
        return norm.pdf(x, loc=self.location, scale=self.scale)

    def normal_cdf(self,x):
        return norm.cdf(x, loc=self.location, scale=self.scale)

    def normal_inv(self,u):
        return norm.ppf(u, loc=self.location, scale=self.scale)

    def truncated_norm_pdf(self,x):
        return self.normal_pdf(x) / (self.normal_cdf(self.b) - self.normal_cdf(self.a))

    def truncated_norm_cdf(self,x):
        return (self.normal_cdf(x) - self.normal_cdf(self.a)) / (self.normal_cdf(self.b) - self.normal_cdf(self.a))

    def rand_truncated_normal(self,u):
        return self.normal_inv((self.normal_cdf(self.b) - self.normal_cdf(self.a)) * u + self.normal_cdf(self.a))


def normal_truncated_neq(alpha_p, location, scale, x):
    par = np.array([location, scale])
    ins = TruncatedNormal(par, -np.inf, 0.0)
    return alpha_p * ins.truncated_norm_cdf(x)

def normal_truncated_pos(alpha_p, location, scale, x):
    par = np.array([location, scale])
    ins = TruncatedNormal(par, 0.0, np.inf)
    return alpha_p * ins.truncated_norm_cdf(x)


def normal_truncated(params , x):
    alpha_neq, loc_neg, scale_neg, alpha_pos, loc_pos, scale_pos = params
    nr = np.zeros_like(x)
    x_negative = x[x<0]
    x_positive = x[x>0]
    nr_negative = normal_truncated_neq(alpha_neq, loc_neg, scale_neg, x_negative)
    nr_positive = params[0] + normal_truncated_pos(alpha_pos, loc_pos, scale_pos, x_positive)
    nr[x<0] = nr_negative
    nr[x>0] = nr_positive
    return nr