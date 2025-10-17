import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

def analyze_1_over_f(csv_path, column="mean_abs", out_dir="results/1overf"):
    df = pd.read_csv(csv_path)
    x = df[column].to_numpy(dtype=float)
    x = x - np.mean(x)

    freqs, psd = welch(x)

    valid = (freqs > 0) & (psd > 0)
    logf, logp = np.log10(freqs[valid]), np.log10(psd[valid])
    slope, intercept, r, _, _ = linregress(logf, logp)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    pd.DataFrame({"freq": freqs, "psd": psd}).to_csv(f"{out_dir}/{base}_psd.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.loglog(freqs[1:], psd[1:], label='PSD')
    plt.loglog(freqs[valid], 10**(intercept + slope*logf), '--', label=f'fit slope={slope:.2f}')
    plt.xlabel("Frequency")
    plt.ylabel("Power spectral density")
    plt.legend()
    plt.title(f"1/f noise analysis ({column})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{base}_psd.png", dpi=150)
    plt.show()

    print(f"{base}: slope={slope:.2f}, R²={r**2:.3f}")
    return slope, r**2

if __name__ == "__main__":
    # single example
    analyze_1_over_f("results\data\mRand_True_pDead_0.5_mInt_3_mDens_0.2_ampSeed_181074_phaseSeed_482152_mSeed_945680_gens_10000_run_1.csv")

    # or batch-process all CSVs
    # import glob
    # for f in glob.glob("results/data/*.csv"):
    #     analyze_1_over_f(f)