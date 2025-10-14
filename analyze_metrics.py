import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyze_runs import INDEX_CSV


def load_run_parquet(path: str) -> np.ndarray:
    '''
    Load parquet file containing simulation data into DataFrame, then convert into 3D array where first dimension is
    generation and last two dimensions are the grid.
    '''
    df = pq.read_table(path, columns=["generation","i","j","p_live","grid_size"], use_threads=True).to_pandas()

    gens = np.sort(df["generation"].unique().astype(int))
    N = int(df["grid_size"].iloc[0]) if "grid_size" in df.columns else int(np.sqrt((df["generation"] == gens[0]).sum()))
    G = len(gens)

    p_live_3d = np.zeros((G, N, N), dtype=np.float32)
    # faster vector fill
    for g_idx, g in enumerate(gens):
        snap = df[df["generation"] == g]
        ii = snap["i"].to_numpy(dtype=int, copy=False)
        jj = snap["j"].to_numpy(dtype=int, copy=False)
        vv = snap["p_live"].to_numpy(dtype=np.float32, copy=False)
        p_live_3d[g_idx, ii, jj] = vv
    
    return p_live_3d


def average_p_live_per_gen(p_live_3d: np.ndarray) -> np.ndarray:
    '''
    Turn 3D array containing the p_live per cell of each generation into 1D array containing average p_live per
    generation.
    '''
    return np.mean(p_live_3d, axis=(1,2))


def stabilised_p_live(data: np.ndarray) -> tuple[int, float, float]:
    '''
    Calculate average p_live of final half of simulation, which is hopefully the value that the simulation stabilised
    on, and the standard deviation. Small standard deviation should be a good indicator of stabilisation.
    '''
    half_size = data.shape[0] // 2
    mean = np.mean(data[half_size:])
    std = np.std(data[half_size:])
    return half_size, mean, std


def simple_plot(data: np.ndarray) -> None:
    plt.plot(range(data.shape[0]), data)
    plt.show()


def plot_generation_data_and_average(data: np.ndarray) -> None:
    '''
    Calculate mean and std from data and plot them. Mean and std are calculated from final half of generations.
    '''
    size = data.shape[0]
    half_size, mean, std = stabilised_p_live(data)
    x = range(size)
    half_x = range(half_size+1, size)
    mean = np.tile(mean, half_size)
    std = np.tile(std, half_size)

    plt.fill_between(half_x, mean-std, mean+std, color='lightgrey', label=f'error: {std[0]:.02}')
    plt.plot(x, data, color='blue', label='data')
    plt.plot(half_x, mean, color='orange', label=f'mean: {mean[0]:.02}')
    plt.legend()
    plt.show()


def process_index() -> pd.DataFrame:
    '''
    Go through all simulations and calculate overall average alive probability (which is hopefully the value that this
    metric stabilises around).
    '''
    index = pd.read_csv(INDEX_CSV)
    simulations = []
    for _, row in index.iterrows():
        _, mean, std = stabilised_p_live(average_p_live_per_gen(load_run_parquet(row['path'])))
        simulations.append({
            "mean": mean,
            "std": std,
            "p_dead": row['p_dead'],
            "meas_density": row['meas_density'],
            "meas_interval": row['meas_interval'],
        })
    return pd.DataFrame(simulations)


def select_metric(data: pd.DataFrame, metrics: tuple) -> np.ndarray:
    '''
    `data` should contain (at least) 5 columns named "mean", "std", "p_dead", "meas_density", and "meas_interval".
    `metrics` should be a tuple of length 3. The first value corresponds to fixing "p_dead", the second to fixing
    "meas_density", the third to fixing "meas_interval". Precisely one of these should be None.

    returns a 2D array where each row is a triple containing the parameter with `None` metrics, the mean and the std for
    each row in `data` where the other two parameters are equal to the given metrics.

    Example: running `group_metrics(data, (0.5, None, 25))` returns the rows in data where p_dead=0.5 and meas_interval=25
    in the format [[meas_density1, mean1, std1], [meas_density2, mean2, std2], etc.].
    '''
    assert (l := len(metrics)) == 3, f"There are 3 parameters, but `specs` has length {l}"
    assert (l := len([val for val in metrics if val is None])) == 1, f"Expected 1 None-value in `specs`, found {l}"

    p_dead, meas_density, meas_interval = metrics
    if p_dead is None:
        fixed_specs = (meas_density, meas_interval)
        values = data.get(["p_dead", "mean", "std", "meas_density", "meas_interval"]).to_numpy()
    elif meas_density is None:
        fixed_specs = (p_dead, meas_interval)
        values = data.get(["meas_density", "mean", "std", "p_dead", "meas_interval"]).to_numpy()
    else:
        fixed_specs = (p_dead, meas_density)
        values = data.get(["meas_interval", "mean", "std", "p_dead", "meas_density"]).to_numpy()
    
    return np.array([row[:-2] for row in values if tuple(row[-2:]) == fixed_specs])


def plot_metric_alone(data: pd.DataFrame, metrics: tuple) -> None:
    '''
    See `select_metrics` for requirements of `metrics`.
    '''
    values = select_metric(data, metrics)

    p_dead, meas_density, meas_interval = metrics
    if p_dead is None:
        xlabel = r"$p_{\text{dead}}$"
        title = f"Initial distribution for meas_density={meas_density:.2}, meas_interval={meas_interval}"
    elif meas_density is None:
        xlabel = r"Measurement density"
        title = f"Measurement density for p_dead={p_dead:.2}, meas_interval={meas_interval}"
    else:
        xlabel = r"Measurement interval (generations)"
        title = f"Measurement interval for p_dead={p_dead:.02}, meas_density={meas_density:.2}"
    
    plt.errorbar(values[:,0], values[:,1], yerr=values[:,2], fmt="-o", capsize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average alive probability")
    plt.show()


def plot_metric_sweep(data: pd.DataFrame, primary: str, secondary: str, third_value: float) -> None:
    '''
    Plots average alive probability against primary for each value of secondary, where the third parameter is fixed.
    '''
    for sec_value in np.unique(data.get([secondary]).to_numpy()):
        if primary == "p_dead":
            xlabel = r"$p_{\text{dead}}$"
            if secondary == "meas_density":
                specs = (None, sec_value, third_value)
                title = f"Initial distribution for meas_interval={third_value}"
            else:
                specs = (None, third_value, sec_value)
                title = f"Initial distribution for meas_density={third_value:.2}"
        elif primary == "meas_density":
            xlabel = "Measurement density"
            if secondary == "p_dead":
                specs = (sec_value, None, third_value)
                title = f"Measurement density for meas_interval={third_value}"
            else:
                specs = (third_value, None, sec_value)
                title = f"Measurement density for p_dead={third_value:.2}"
        else:
            xlabel = "Measurement interval (generations)"
            if secondary == "p_dead":
                specs = (sec_value, third_value, None)
                title = f"Measurement interval for meas_density={third_value:.2}"
            else:
                specs = (third_value, sec_value, None)
                title = f"Measurement interval for p_dead={third_value:.2}"
        values = select_metric(data, specs)
        
        plt.errorbar(values[:,0], values[:,1], yerr=values[:,2], fmt="-o", capsize=3, label=f"{secondary}={sec_value}")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average alive probability")
    plt.legend()
    plt.show()


def main():
    # index = pd.read_csv(INDEX_CSV)
    # run = load_run(index.at[0, "path"])
    # data = average_p_live_per_gen(run)
    # print(stabilised_p_live(data))

    simulations = process_index()
    # plot_metric_alone(simulations, (None, 0.1, 5))
    plot_metric_sweep(simulations, "meas_density", "meas_interval", 0.6)
    

if __name__ == '__main__':
    main()
