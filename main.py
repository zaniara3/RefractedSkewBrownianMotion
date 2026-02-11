import argparse
import numpy as np
from pathlib import Path

from sbm_truncnorm.sbm import SkewBrownianMotion
from sbm_truncnorm.fit import fit_truncated_normal_to_sbm_cdf
from sbm_truncnorm.truncnorm import normal_truncated
from sbm_truncnorm.sampling import sample_mixture_truncnorm
from sbm_truncnorm.utils import plot_hist_vs_pdf, plot_cdf_theory_vs_emp

def build_grid():
    z_val1 = np.linspace(-8, 8, 2000)
    z_val2 = np.linspace(-0.5, 0.5, 200)
    x = np.concatenate((z_val1, z_val2))
    x.sort()
    if np.where(x == 0)[0].size > 0:
        raise ValueError("there is zero in x")
    return x

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Mn", type=float, default=-0.1)
    p.add_argument("--Mp", type=float, default=0.1)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = build_grid()

    sbm_params = [args.Mn, args.Mp, args.t, args.beta]
    sbm_model = SkewBrownianMotion(sbm_params)

    sbm_pdf = sbm_model.pdf_conv_func(x)
    sbm_cdf = sbm_model.cdf_conv_func(x)

    params_full, res_neg, res_pos = fit_truncated_normal_to_sbm_cdf(
        x, sbm_cdf, args.Mn, args.Mp, args.t, args.beta
    )

    # Estimated truncated-normal mixture CDF on grid (if you want it)
    tr_est_cdf = normal_truncated(params_full, x)

    sample = sample_mixture_truncnorm(params_full, n=args.n, seed=args.seed)

    # CDF at sorted sample using SBM theoretical CDF
    sample_sorted = np.sort(sample)
    sbm_cdf_sample_sorted = sbm_model.cdf_conv_func(sample_sorted)

    plot_hist_vs_pdf(
        sample, x, sbm_pdf,
        outpath=outdir / "hist_vs_pdf.png"
    )
    plot_cdf_theory_vs_emp(
        sample, sbm_cdf_sample_sorted,
        outpath=outdir / "cdf_theory_vs_emp.png"
    )

    # Save fitted params
    np.savetxt(outdir / "fitted_params.txt", params_full, header="alpha_neg loc_neg scale_neg alpha_pos loc_pos scale_pos")

    print("Fit done.")
    print("params_full =", params_full)
    print("neg success:", res_neg.success, "| pos success:", res_pos.success)

if __name__ == "__main__":
    main()
