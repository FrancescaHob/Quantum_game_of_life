"""
batch_runner.py
Runs multiple simulations of the Quantum Game of Life with varying parameters,
records results to Parquet files, and maintains an index CSV for easy reference.
Includes optional live visualization using Pygame and Matplotlib.
Stores data in a structured directory format based on parameters.
"""
import os
import time
import numpy as np
import pandas as pd
import pygame
import matplotlib.pyplot as plt

# from main import (
#     GRID_SIZE, CELL_SIZE, make_random_grid, update_grid,
#     display_grid, measurement
# )

import main as core

# ========= Fixed grid size for this batch =========
GRID_SIZE_RUN = 20
TARGET_WINDOW_PX = 800  # desired total window width/height in pixels

# make sure it's an integer >= 1
core.CELL_SIZE = max(1, int(TARGET_WINDOW_PX // GRID_SIZE_RUN))

# ========= Batch settings =========
LIVE_VIEW            = True                   # Toggle live visualization (True/False) ps: triggy to close with ctrl+c if True
P_DEAD_VALUES        = [0.1, 0.2]             # initial dead probability values to sweep (can take a list)
SEED_AMP_RANGE       = range(12348, 12349)    # amplitude seeds (excluded, so for range(1,2) only 1)
SEED_PHASE_RANGE     = range(54321, 54322)    # phase seeds (excluded)
MEASURE_INTERVALS    = [5, 10]                # measurement steps (can take a list)
MEASURE_DENSITIES    = [0.3, 0.6]             # fraction of cells measured (can take a list)
N_RERUNS             = 1                      # repeats per parameter combo
MAX_GENERATIONS      = 50                     # target steps per run
FPS_CAP              = 50                     # UI refresh rate

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
    os.makedirs(dir_path, exist_ok=True)
    existing = [f for f in os.listdir(dir_path) if f.startswith("run")]
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

# ========= Save helpers =========
def save_entropy_csv(dir_path, run_id, entropy_values):
    path = os.path.join(dir_path, f"run{run_id}_entropy.csv")
    gens = np.arange(1, len(entropy_values) + 1, dtype=int)
    pd.DataFrame({"generation": gens, "entropy_bits": entropy_values}).to_csv(path, index=False)
    return path

def save_parquet_and_index(df, dir_path, run_id,
                           p_dead_init, m_density, m_interval,
                           seed_amp, seed_phase, grid_size,
                           entropy_values):
    os.makedirs(dir_path, exist_ok=True)

    gens_total = int(df["generation"].max())
    rows_total = len(df)

    parquet_path = os.path.join(dir_path, f"run{run_id}_generations{gens_total}.parquet")
    df.to_parquet(parquet_path, index=False)

    entropy_csv_path = save_entropy_csv(dir_path, run_id, entropy_values)

    # summaries for index
    final_entropy = float(entropy_values[-1]) if entropy_values else np.nan
    tail = np.array(entropy_values[-min(20, len(entropy_values)):]) if entropy_values else np.array([])
    tail_var = float(tail.var()) if tail.size else np.nan

    index_path = os.path.join(OUT_DIR, "index.csv")
    new_row = pd.DataFrame([{
        "path": parquet_path,
        "entropy_csv": entropy_csv_path,
        "grid_size": int(grid_size),            # <— include grid_size in catalog
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
    }])

    if not os.path.exists(index_path):
        new_row.to_csv(index_path, index=False, mode="w")
    else:
        new_row.to_csv(index_path, index=False, mode="a", header=False)

    return parquet_path, gens_total, rows_total, entropy_csv_path

# ========= Main =========
def main():
    t0 = time.time()
    saved_paths = []

    for p_dead_init in P_DEAD_VALUES:
        for m_density in MEASURE_DENSITIES:
            for m_interval in MEASURE_INTERVALS:
                for seed_amp in SEED_AMP_RANGE:
                    for seed_phase in SEED_PHASE_RANGE:
                        for _ in range(N_RERUNS):
                            dir_path = get_dir_path(GRID_SIZE_RUN, p_dead_init, m_density, m_interval, seed_amp, seed_phase)
                            run_id = get_next_run_id(dir_path)

                            print(f"Running N={GRID_SIZE_RUN}, pDead={p_dead_init}, D={m_density}, I={m_interval}, "
                                  f"A={seed_amp}, P={seed_phase}, run={run_id} "
                                  f"({'live' if LIVE_VIEW else 'batch'})")

                            if LIVE_VIEW:
                                # Create initial grid with fixed size and run live
                                grid0 = core.make_random_grid(seed_amp, seed_phase, p_dead=p_dead_init, grid_size=GRID_SIZE_RUN)
                                df, gens_total, entropy_values = run_with_live_view_and_record(
                                    grid0, m_interval, m_density,
                                    seed_amp, seed_phase, p_dead_init, run_id
                                )
                            else:
                                df, gens_total, entropy_values = run_batch_and_record(
                                    seed_amp, seed_phase, m_interval, m_density, p_dead_init, run_id, GRID_SIZE_RUN
                                )

                            parquet_path, gens_total, rows_total, entropy_csv_path = save_parquet_and_index(
                                df, dir_path, run_id,
                                p_dead_init, m_density, m_interval, seed_amp, seed_phase, GRID_SIZE_RUN,
                                entropy_values
                            )

                            saved_paths.append(parquet_path)
                            print(f"  -> saved {parquet_path} (+ {os.path.basename(entropy_csv_path)}) "
                                  f"(gens {gens_total}, rows {rows_total:,})")

    dt = time.time() - t0
    print("\nDone.")
    for p in saved_paths:
        print("•", p)
    print(f"\nTotal runs: {len(saved_paths)}, total time: {dt:.2f}s")

if __name__ == "__main__":
    main()
