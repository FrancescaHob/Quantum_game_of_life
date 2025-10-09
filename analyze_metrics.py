import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyze_runs import INDEX_CSV


def load_run(path: str) -> np.ndarray:
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


def plot_data_and_average(data: np.ndarray) -> None:
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
        _, mean, std = stabilised_p_live(average_p_live_per_gen(load_run(row['path'])))
        simulations.append({
            "mean": mean,
            "std": std,
            "p_dead": row['p_dead'],
            "meas_density": row['meas_density'],
            "meas_interval": row['meas_interval'],
        })
    return pd.DataFrame(simulations)


def main():
    # index = pd.read_csv(INDEX_CSV)
    # run = load_run(index.at[0, "path"])
    # data = average_p_live_per_gen(run)
    # print(stabilised_p_live(data))

    simulations = process_index()
    print(simulations)

if __name__ == '__main__':
    main()
