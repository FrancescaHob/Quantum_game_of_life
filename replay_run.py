import sys
import numpy as np
import pyarrow.parquet as pq
import pygame

import main as core  # import the module

TARGET_WINDOW_PX = 600  # desired window width/height in pixels (square)

def load_frames(parquet_path):
    cols = ["generation","i","j","live_real","live_imag","dead_real","dead_imag","grid_size"]
    table = pq.read_table(parquet_path, columns=cols)
    df = table.to_pandas()

    gens = sorted(df["generation"].unique())
    N = int(df["grid_size"].iloc[0])

    frames = []
    for g in gens:
        snap = df[df["generation"] == g]
        grid = np.empty((N, N), dtype=object)
        ii = snap["i"].to_numpy(int, copy=False)
        jj = snap["j"].to_numpy(int, copy=False)
        live = snap["live_real"].to_numpy(float) + 1j * snap["live_imag"].to_numpy(float)
        dead = snap["dead_real"].to_numpy(float) + 1j * snap["dead_imag"].to_numpy(float)
        for k in range(len(snap)):
            grid[ii[k], jj[k]] = np.array([live[k], dead[k]], dtype=complex)
        frames.append(grid)
    return frames, N

def run_pygame(frames, N):
    # scale cells so the window stays ~TARGET_WINDOW_PX
    core.CELL_SIZE = max(1, int(TARGET_WINDOW_PX // N))

    pygame.init()
    screen = pygame.display.set_mode((N * core.CELL_SIZE, N * core.CELL_SIZE))
    pygame.display.set_caption(f"Replay — N={N}, cell={core.CELL_SIZE}px")
    clock = pygame.time.Clock()
    idx, running = 0, True

    # initial draw
    core.display_grid(screen, frames[idx])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        core.display_grid(screen, frames[idx])  # uses updated core.CELL_SIZE
        pygame.display.flip()
        clock.tick(20)  # fps
        idx = (idx + 1) % len(frames)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_run.py <path_to_parquet>")
        sys.exit(1)
    path = sys.argv[1]
    frames, N = load_frames(path)
    run_pygame(frames, N)
