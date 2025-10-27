#!/usr/bin/env python3
"""
plot_density_max_run_debug_save.py

Scans a folder of CSV result files, extracts density and run number,
finds the highest run for each parameter-combination (everything except run),
plots density (x) vs highest run number (y), and saves the extracted results
to a CSV file for each (pDead, interval) combination.

Output filename example:
  phase_transition_pDead_0.4_interval_3_density_0.3-0.4.csv
"""

import os
import re
import sys
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

# === Config ===
base_folder = "results/results-23-10-2025/results/data/all/"  # modify if needed
output_folder = "results/phase_transition_outputs/"
os.makedirs(output_folder, exist_ok=True)

# === Helper regexes ===
re_density = re.compile(r"density_([0-9]*\.?[0-9]+)")
re_run = re.compile(r"_run_([0-9]+)")
re_pDead = re.compile(r"pDead_([0-9]*\.?[0-9]+)")
re_interval = re.compile(r"interval_([0-9]+)")

def main():
    if not os.path.exists(base_folder):
        print(f"ERROR: base_folder does not exist: {base_folder!r}")
        sys.exit(1)

    filenames = sorted(os.listdir(base_folder))
    print(f"Total files in folder: {len(filenames)}")

    # Store max run per (pDead, interval, other parameters)
    max_runs = defaultdict(lambda: defaultdict(int))

    for fname in filenames:
        if os.path.isdir(os.path.join(base_folder, fname)):
            continue

        dmatch = re_density.search(fname)
        rmatch = re_run.search(fname)
        pmatch = re_pDead.search(fname)
        imatch = re_interval.search(fname)

        if not dmatch or not rmatch or not pmatch or not imatch:
            continue

        density = float(dmatch.group(1))
        runnum = int(rmatch.group(1))
        pDead = float(pmatch.group(1))
        interval = int(imatch.group(1))

        key = re_run.sub("_run_", fname)
        combo_key = (pDead, interval)

        # Update maximum run for this parameter combination within (pDead, interval)
        max_runs[combo_key][key] = max(max_runs[combo_key][key], runnum)

    if not max_runs:
        print("No matching files found. Exiting.")
        return

    # === Plot each (pDead, interval) combination and save data ===
    for (pDead, interval), runs_dict in sorted(max_runs.items()):
        densities = []
        highest_runs = []

        for key, run_val in runs_dict.items():
            d2 = re_density.search(key)
            if not d2:
                continue
            densities.append(float(d2.group(1)))
            highest_runs.append(run_val)

        if not densities:
            continue

        paired = sorted(zip(densities, highest_runs), key=lambda x: x[0])
        xs, ys = zip(*paired)

        # Save results to CSV
        min_d, max_d = min(xs), max(xs)
        csv_filename = f"phase_transition_pDead_{pDead}_interval_{interval}_density_{min_d}-{max_d}.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["density", "highest_run"])
            writer.writerows(zip(xs, ys))

        print(f"Saved: {csv_path}")

        # Plot and save figure
        plt.figure(figsize=(9, 6))
        plt.scatter(xs, ys, s=80, edgecolor='k',marker= '.', alpha=0.7)
        plt.title(f"Number of runs until simulation staus alive for 2500 generations\n"
                  f"(pDead={pDead}, interval={interval})")
        plt.xlabel("Measurement density")
        plt.ylabel("Number of runs (max. 25 reruns)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_filename = csv_filename.replace('.csv', '.png')
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
