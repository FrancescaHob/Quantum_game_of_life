import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def make_scatter(df):
    """
    Create 5 scatter plots (for pDead = 0, 0.2, 0.4, 0.6, 0.8),
    showing interval (log scale) vs density, with color by class.
    Only one legend is displayed outside the plots.
    """
    pdead_values = [0, 0.2, 0.4, 0.6, 0.8]
    colors = {0: 'blue', 1: 'orange', 2: 'green'}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    # We'll collect handles manually (one per class)
    scatter_handles = []
    scatter_labels = []
    classes = []

    for ax, pDead in zip(axes, pdead_values):
        subset = df[df['pDead'] == pDead]
        if subset.empty:
            ax.set_title(f"pDead = {pDead}\n(no data)")
            ax.axis('off')
            continue

        for cls, color in colors.items():
            sub = subset[subset['class'] == cls]
            sc = ax.scatter(sub['interval'], sub['density'],
                            color=color, alpha=0.7, label=f'class {cls}')

            # Only collect one handle per class (first time only)
            if not cls in classes:
                scatter_handles.append(sc)
                classes.append(cls)
                match cls:
                    case 0:
                        scatter_labels.append("Stayed alive (first time)")
                    case 1:
                        scatter_labels.append("Died (25 times)")
                    case 2:
                        scatter_labels.append("Stayed alive after few reruns")

        ax.set_xscale('log')
        ax.set_title(f"pDead = {pDead}")
        ax.set_xlabel('Interval (log scale)')
        ax.set_ylabel('Density')

    fig.legend(scatter_handles, scatter_labels)

    plt.tight_layout(rect=(0, 0, 0.9, 1))  # Leave space for legend
    plt.show()

def build_df(folder_path=r"results/results_1_used_for_making_initial_goldilock_zone/plots/finished/life_probability"):
    """
    Build a pandas DataFrame from filenames containing encoded simulation parameters.

    Each filename is expected to follow a pattern like:
    'grid_50_pDead_0.2_interval_1_density_0.1_run_1_gens_2500_ampSeed_390860_phaseSeed_493018_mSeed_249581.png'

    The resulting DataFrame contains:
      - pDead
      - interval
      - density
      - class (0 for run=1, 1 for run=25, otherwise 2)
    """

    pattern = re.compile(
        r"pDead_(?P<pDead>[\d.]+)_interval_(?P<interval>[\d.]+)_density_(?P<density>[\d.]+)_run_(?P<run>\d+)"
    )

    data = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".png"):
            continue

        match = pattern.search(fname)
        if match:
            pDead = float(match.group("pDead"))
            interval = float(match.group("interval"))
            density = float(match.group("density"))
            run = int(match.group("run"))

            # Determine class
            if run == 1:
                cls = 0
            elif run == 25:
                cls = 1
            else:
                cls = 2

            data.append({
                "pDead": pDead,
                "interval": interval,
                "density": density,
                "class": cls
            })

    df = pd.DataFrame(data)
    return df

def main():
    df = build_df()
    make_scatter(df)

if __name__ == '__main__':
    main()