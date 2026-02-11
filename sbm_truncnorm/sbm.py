import numpy as np
from scipy import integrate
from scipy.special import erfc


def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))


def h_pdf(t, x, mu):
    return np.abs(x) / np.sqrt(2 * np.pi * t ** 3) * np.exp(-(x + mu * t) ** 2 / (2 * t))


def norm_pdf(t, y, mu):
    return 1 / np.sqrt(2 * np.pi * t) * np.exp(-(y - mu * t) ** 2 / (2 * t))


class SkewBrownianMotion():
    def __init__(self, params):
        self.Mn = params[0]
        self.Mp = params[1]
        self.t = params[2]
        self.beta = params[3]

    def pdf_sbm_general(self, b, tau, y):

        if y > 0:
            pdf = 2 * (1 + self.beta) * np.exp(2 * self.Mp * y) \
                  * h_pdf(self.t - tau, (1 + self.beta) * b + y, self.Mp) \
                  * h_pdf(tau, (1 - self.beta) * b, -self.Mn)
        else:
            pdf = 2 * (1 - self.beta) * np.exp(2 * self.Mn * y) \
                  * h_pdf(self.t - tau, (1 + self.beta) * b, self.Mp) \
                  * h_pdf(tau, (1 - self.beta) * b - y, -self.Mn)
        return pdf


    def cdf_sbm_general(self, b, tau, y):
        if y < 0:
            cdf = 2 * (1 - self.beta) * safe_exp(2 * self.Mn * (1 - self.beta) * b) * h_pdf(self.t - tau, (1 + self.beta) * b, self.Mp) * (
                        norm_pdf(tau, (1 - self.beta) * b - y, -self.Mn) - 0.5 * self.Mn *
                          erfc(((1 - self.beta) * b - y + self.Mn * (tau)) / (np.sqrt(2 * (tau)))))
        else:
            cdf = 2 * (1 + self.beta) * safe_exp(-2 * self.Mp * (1 + self.beta) * b) * h_pdf(self.t - tau, (1 - self.beta) * b, -self.Mn) * (
                        norm_pdf(tau, (1 + self.beta) * b + y, self.Mp)+ 0.5 * self.Mp *
                             erfc(((1 + self.beta) * b + y - self.Mp * tau) / (np.sqrt(2 * tau))))
        return cdf

    def pdf_conv_func(self, y):
        pdf = np.array([integrate.dblquad(lambda b,tau: self.pdf_sbm_general(b,tau, y[i]),0,self.t,0,np.inf)[0] for i in range(y.shape[0])])
        return pdf

    def cdf_conv_func(self,y):
        cdf = np.zeros(y.shape)
        y_negative = y[y<0]
        y_positive = y[y>0]
        cdf_neg = np.array([integrate.dblquad(lambda b,tau : self.cdf_sbm_general(b,tau, y_negative[i]),0,self.t,0,np.inf)[0] for i in range(y_negative.shape[0])])
        cdf_pos = 1- np.array([integrate.dblquad(lambda b,tau : self.cdf_sbm_general(b,tau, y_positive[i]),0,self.t,0,np.inf)[0] for i in range(y_positive.shape[0])])
        cdf[y<0] = cdf_neg
        cdf[y>0] = cdf_pos
        return cdf

    # Empirical cdf
    def cdf_emp(self, z_values , pdf_sbm, method):
        cdf_x = np.zeros_like(z_values)
        if method == 'trapz':
            cdf_x[0] = 0
            cdf_x[1:] = np.array([np.trapz(pdf_sbm[0:i], x=z_values[0:i]) for i in range(1,z_values.shape[0])])
        if method == 'simpson':
            cdf_x[0:2] = 0
            cdf_x[2:] = np.array([integrate.simpson(pdf_sbm[0:i], x=z_values[0:i]) for i in range(2,z_values.shape[0])])
        return cdf_x
