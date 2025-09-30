"""
batch_runner.py
Runs multiple simulations of the Quantum Game of Life with varying parameters,
records results to Parquet files, and maintains an index CSV for easy reference.
Uses multiprocessing for parallel execution.
Includes optional live visualization using Pygame and Matplotlib (disabled in parallel mode).
Stores data in a structured directory format based on parameters.
"""
import os
import time
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp

# from main import (
#     GRID_SIZE, CELL_SIZE, make_random_grid, update_grid,
#     display_grid, measurement
# )

import main as core

# ========= Fixed grid size for this batch =========
GRID_SIZE_RUN = 150
TARGET_WINDOW_PX = 800  # desired total window width/height in pixels

# make sure it's an integer >= 1
core.CELL_SIZE = max(1, int(TARGET_WINDOW_PX // GRID_SIZE_RUN))

# ========= Batch settings =========
LIVE_VIEW            = False                  # Toggle live visualization (True/False) ps: triggy to close with ctrl+c if True
P_DEAD_VALUES        = [0.8, 0.6]                  # initial dead probability values to sweep (can take a list)
SEED_AMP_RANGE       = range(12348, 12349)    # amplitude seeds (excluded, so for range(1,2) only 1)
SEED_PHASE_RANGE     = range(54321, 54322)    # phase seeds (excluded)
MEASURE_INTERVALS    = [5]                    # measurement steps (can take a list)
MEASURE_DENSITIES    = [0.1]                  # fraction of cells measured (can take a list)
N_RERUNS             = 4                     # repeats per parameter combo
MAX_GENERATIONS      = 50                     # target steps per run
FPS_CAP              = 50                     # UI refresh rate
N_WORKERS            = None                   # Number of worker processes (None uses all CPU cores - 1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "simulation_data")

# ========= Helpers =========
def fmt_num(x):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x).replace(".", "p")

def get_dir_path(grid_size, p_dead_init, m_density, m_interval, seed_amp, seed_phase):
    """Folder structure: N{grid}/pDead.../measD.../measI.../seedA...-seedP..."""
    return os.path.join(
        str(OUT_DIR),
        f"N{int(grid_size)}",
        f"pDead{fmt_num(p_dead_init)}",
        f"measD{fmt_num(m_density)}",
        f"measI{fmt_num(m_interval)}",
        f"seedA{fmt_num(seed_amp)}-seedP{fmt_num(seed_phase)}"
    )

def get_next_run_id(dir_path):
    # CRITICAL: This function must be called only in the single-threaded main process
    # to avoid race conditions. It creates the directory and returns the next ID.
    os.makedirs(dir_path, exist_ok=True)
    existing = [f for f in os.listdir(dir_path) if f.startswith("run") and f.endswith(".parquet")]
    # Since run IDs are 1-indexed, the length of existing runs is the next run ID
    return len(existing) + 1

# ========= Collapsed 2×2 block entropy (periodic edges) =========
def collapsed_block_entropy_bits(p_live_grid: np.ndarray):
    """
    Collapse cell to {0,1} via p_live>=0.5, then Shannon entropy (bits) of 2x2 block codes
    over all positions with wrap-around. Returns H in [0,4].
    """
    b = (p_live_grid >= 0.5).astype(np.uint8)
    a  = b
    r  = np.roll(b, -1, axis=1)
    d  = np.roll(b, -1, axis=0)
    dr = np.roll(np.roll(b, -1, axis=0), -1, axis=1)
    code   = (a << 3) | (r << 2) | (d << 1) | dr   # 0..15
    counts = np.bincount(code.ravel(), minlength=16).astype(np.float64)
    total  = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p > 0
    H = -(p[nz] * np.log2(p[nz])).sum()
    return float(H)

# ========= Recording helper =========
def grid_to_records(grid, gen, seed_amp, seed_phase,
                    m_interval, m_density, p_dead_init,
                    run_id, gens_total):
    N = len(grid)
    recs = []
    for i in range(N):
        for j in range(N):
            live, dead = grid[i][j]
            recs.append({
                "generation": gen,
                "i": i,
                "j": j,
                "live_real": float(np.real(live)),
                "live_imag": float(np.imag(live)),
                "dead_real": float(np.real(dead)),
                "dead_imag": float(np.imag(dead)),
                "p_live": float(np.abs(live)**2),
                "phi_live": float(np.angle(live)),
                # metadata
                "seed_amp": int(seed_amp),
                "seed_phase": int(seed_phase),
                "grid_size": int(N),
                "p_dead": float(p_dead_init),
                "measure_interval": int(m_interval),
                "measure_density": float(m_density),
                "max_generations": int(gens_total),
                "run_id": int(run_id),
                "runner_version": "v13",
            })
    return recs

# ========= Live visualization (Heatmap + Entropy) =========
def run_with_live_view_and_record(grid, m_interval, m_density,
                                  seed_amp, seed_phase,
                                  p_dead_init, run_id):
    """
    Live Pygame + Matplotlib dashboard AND full recording of each generation.
      Left: cumulative average p(live) heatmap
      Right: collapsed 2×2 block entropy (bits) over time
    Close Pygame to stop early; we save what’s done with correct generation count.
    """
    # Pygame
    pygame.init()
    N = len(grid)
    screen = pygame.display.set_mode((N * core.CELL_SIZE, N * core.CELL_SIZE))

    # Matplotlib
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    heatmap = axes[0].imshow(np.zeros((N, N)), cmap='viridis', animated=True)
    plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04, label='Average p(live)')
    axes[0].set_title('Average Cell Activity (cumulative)')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')

    ent_line, = axes[1].plot([], [], lw=2)
    axes[1].set_title("Collapsed 2×2 block entropy")
    axes[1].set_xlim(0, MAX_GENERATIONS)
    axes[1].set_ylim(0, 4.0)
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Entropy (bits)")
    fig.tight_layout()

    total_activity = np.zeros((N, N))
    entropy_values = []

    all_rows = []
    gens_total = MAX_GENERATIONS
    # gen 0
    all_rows.extend(grid_to_records(
        grid, 0, seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, gens_total
    ))

    generation = 0
    running = True

    while running and generation < MAX_GENERATIONS:
        grid = core.update_grid(grid)
        generation += 1

        if generation % m_interval == 0:
            grid = core.measurement(grid, measurement_density=m_density)

        # record full grid
        all_rows.extend(grid_to_records(
            grid, generation, seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, gens_total
        ))

        # p_live grid + accumulators
        p_live_grid = np.empty((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                p_live = abs(grid[i][j][0]) ** 2
                p_live_grid[i, j] = p_live
                total_activity[i, j] += p_live

        # entropy
        H_bits = collapsed_block_entropy_bits(p_live_grid)
        entropy_values.append(H_bits)

        # ---- UI ----
        core.display_grid(screen, grid)
        pygame.display.flip()

        heatmap.set_array(total_activity / max(1, generation))
        ent_line.set_data(np.arange(1, len(entropy_values) + 1), entropy_values)
        axes[1].set_xlim(0, max(10, generation))

        plt.pause(1 / FPS_CAP)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    gens_total = generation
    pygame.quit()
    plt.ioff()
    plt.close()

    return pd.DataFrame.from_records(all_rows), gens_total, entropy_values

# ========= Batch/no-UI run (records data) =========
def run_batch_and_record(seed_amp, seed_phase, m_interval, m_density,
                         p_dead_init, run_id, grid_size):
    grid = core.make_random_grid(seed_amp, seed_phase, p_dead=p_dead_init, grid_size=grid_size)
    N = len(grid)
    all_rows = []
    entropy_values = []

    gens_total = MAX_GENERATIONS
    # gen 0
    all_rows.extend(grid_to_records(
        grid, 0, seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, gens_total
    ))

    for gen in range(1, MAX_GENERATIONS + 1):
        grid = core.update_grid(grid)
        if gen % m_interval == 0:
            grid = core.measurement(grid, measurement_density=m_density)

        all_rows.extend(grid_to_records(
            grid, gen, seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, gens_total
        ))

        # entropy
        p_live_grid = np.empty((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                p_live_grid[i, j] = abs(grid[i][j][0]) ** 2
        entropy_values.append(collapsed_block_entropy_bits(p_live_grid))

    return pd.DataFrame.from_records(all_rows), gens_total, entropy_values

# ========= Save helpers (Modified to only save files and return index data) =========
def save_entropy_csv(dir_path, run_id, entropy_values):
    path = os.path.join(dir_path, f"run{run_id}_entropy.csv")
    gens = np.arange(1, len(entropy_values) + 1, dtype=int)
    pd.DataFrame({"generation": gens, "entropy_bits": entropy_values}).to_csv(path, index=False)
    return path

def save_parquet_and_get_index_data(df, dir_path, run_id,
                           p_dead_init, m_density, m_interval,
                           seed_amp, seed_phase, grid_size,
                           entropy_values):
    """Saves Parquet and Entropy CSV files and returns a dictionary for the index row."""
    # Note: os.makedirs(dir_path) is now safely called in the single-threaded main function.

    gens_total = int(df["generation"].max())
    rows_total = len(df)

    parquet_path = os.path.join(dir_path, f"run{run_id}_generations{gens_total}.parquet")
    df.to_parquet(parquet_path, index=False)

    entropy_csv_path = save_entropy_csv(dir_path, run_id, entropy_values)

    # summaries for index
    final_entropy = float(entropy_values[-1]) if entropy_values else np.nan
    tail = np.array(entropy_values[-min(20, len(entropy_values)):]) if entropy_values else np.array([])
    tail_var = float(tail.var()) if tail.size else np.nan

    # Return the dictionary to be aggregated by the main process
    return {
        "path": parquet_path,
        "entropy_csv": entropy_csv_path,
        "grid_size": int(grid_size),
        "p_dead": p_dead_init,
        "meas_density": m_density,
        "meas_interval": m_interval,
        "seed_amp": seed_amp,
        "seed_phase": seed_phase,
        "run_id": run_id,
        "generations": gens_total,
        "rows": rows_total,
        "final_entropy_bits": final_entropy,
        "tail_var_entropy": tail_var,
    }, parquet_path, gens_total, rows_total

# ========= Worker Function for Multiprocessing (Top-Level) =========
def run_single_task(params):
    """
    Executes one simulation run using the pre-calculated unique run_id.
    Returns data for the global index.
    """
    try:
        # Unpack parameters, including the unique run_id and grid_size
        p_dead_init, m_density, m_interval, seed_amp, seed_phase, run_id, grid_size = params

        # Recalculate dir_path using the received parameters
        dir_path = get_dir_path(grid_size, p_dead_init, m_density, m_interval, seed_amp, seed_phase)

        print(f"Running N={grid_size}, pDead={p_dead_init}, D={m_density}, I={m_interval}, "
              f"A={seed_amp}, P={seed_phase}, run={run_id} (batch/worker)")

        # Always run batch mode when using multiprocessing
        df, gens_total, entropy_values = run_batch_and_record(
            seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, grid_size
        )

        # Save files using the unique run_id and return index data
        index_data, parquet_path, gens_total, rows_total = save_parquet_and_get_index_data(
            df, dir_path, run_id,
            p_dead_init, m_density, m_interval, seed_amp, seed_phase, grid_size,
            entropy_values
        )

        # Return the index row data and path for final logging/index aggregation
        return index_data, parquet_path, rows_total

    except Exception as e:
        # Handle errors gracefully in the worker process
        print(f"ERROR: Task {params} failed with error: {e}")
        return None, None, 0

# ========= Main (Multiprocessing Implementation) =========
def main():
    t0 = time.time()

    # Check for live view when running in parallel (only possible with 1 worker)
    workers = N_WORKERS if N_WORKERS is not None else mp.cpu_count() - 1
    # if workers > 1 and LIVE_VIEW:
    #     print(f"WARNING: LIVE_VIEW disabled. Cannot run Pygame visualization across {workers} processes.")
    #     LIVE_VIEW = False

    # 1. Create the mesh of primary parameters (NO RERUNS in the mesh)
    primary_param_mesh = itertools.product(
        P_DEAD_VALUES,      # 1. p_dead_init
        MEASURE_DENSITIES,  # 2. m_density
        MEASURE_INTERVALS,  # 3. m_interval
        SEED_AMP_RANGE,     # 4. seed_amp
        SEED_PHASE_RANGE,   # 5. seed_phase
    )

    task_list = []

    # 2. Iterate and pre-calculate unique run_ids in the single-threaded main process
    for p_dead_init, m_density, m_interval, seed_amp, seed_phase in primary_param_mesh:

        # Calculate the base directory and create it if necessary. This prevents race conditions.
        dir_path = get_dir_path(GRID_SIZE_RUN, p_dead_init, m_density, m_interval, seed_amp, seed_phase)

        # Determine the starting run ID that accounts for existing runs in this directory
        start_run_id = get_next_run_id(dir_path)

        for run_index in range(N_RERUNS):
            unique_run_id = start_run_id + run_index

            # Create a task tuple for each unique simulation run
            task_list.append((
                p_dead_init,
                m_density,
                m_interval,
                seed_amp,
                seed_phase,
                unique_run_id, # <-- The guaranteed unique run ID
                GRID_SIZE_RUN  # <-- Passing the global grid size
            ))

    print(f"Starting {len(task_list)} simulation tasks using {workers} worker(s)...")

    results = []

    if workers == 1 or LIVE_VIEW:
        # Run sequentially if only 1 worker or LIVE_VIEW is enabled
        print("Running in sequential mode...")
        for params in task_list:
            result = run_single_task(params)
            if result:
                results.append(result)
    else:
        # Run in parallel mode
        with mp.Pool(processes=workers) as pool:
            # Use pool.map to distribute the task list to run_single_task
            results = pool.map(run_single_task, task_list)

    # Filter out failed tasks (which returned None)
    successful_results = [r for r in results if r is not None]

    # --- Index Aggregation and Final Write ---

    if successful_results:
        # Unpack collected data
        index_rows_data = [data for data, path, rows in successful_results]
        saved_paths = [path for data, path, rows in successful_results]

        final_index_df = pd.DataFrame.from_records(index_rows_data)

        index_path = os.path.join(OUT_DIR, "index.csv")

        # Write the final index file safely once
        mode = "w"
        header = True
        if os.path.exists(index_path):
            mode = "a"
            header = False

        final_index_df.to_csv(index_path, index=False, mode=mode, header=header)

        dt = time.time() - t0
        print("\nDone.")
        for p in saved_paths:
            print("•", p)
        print(f"\nTotal runs: {len(successful_results)}, total time: {dt:.2f}s")

    else:
        dt = time.time() - t0
        print(f"\nNo successful runs completed. Total time: {dt:.2f}s")


# Ensure this block is used for multiprocessing safety on some platforms
if __name__ == '__main__':
    main()
