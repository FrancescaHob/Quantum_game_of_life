import os
from main import SEED_MEASUREMENT, SEED_PHASE, make_random_grid, update_grid, measurement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

generations = 50
rng_phase = np.random.default_rng(SEED_PHASE)
rng_measurement = np.random.default_rng(SEED_MEASUREMENT)

def run_batch(p_deads, measurement_intervals, measurement_densities, random_measurements):
    grid_size = 50
    for random_measurement in random_measurements:
        for p_dead in p_deads:
            for measurement_interval in measurement_intervals:
                for measurement_density in measurement_densities:
                    run = 1

                    while True:
                        # generate new random seeds
                        seed_amplitude = random.randint(1, 999999)
                        seed_phase = random.randint(1, 999999)
                        seed_measurement = random.randint(1, 999999)

                        print(f"--- Run {run} ---")
                        print(f"Seeds: amp={seed_amplitude}, phase={seed_phase}, measure={seed_measurement}")

                        # start timer
                        start_time = time.time()

                        # run simulation
                        rerun = run_simulation(seed_amplitude, seed_phase, seed_measurement,
                                               p_dead, measurement_density, grid_size,
                                               random_measurement, measurement_interval, run)

                        # stop timer
                        elapsed = time.time() - start_time
                        print(f"Run {run} completed in {elapsed:.2f} seconds.")

                        if rerun:
                            run += 1
                            print("Rerun requested — starting next run.\n")
                            continue  # go back to the top (new seeds, next run)
                        else:
                            print("Simulation finished successfully.")
                            break  # stop looping

def run_simulation(seed_amplitude, seed_phase, seed_measurement,
                   p_dead, measurement_density, grid_size,
                   random_measurement, measure_interval, run):
    # array to store gen,mean_abs, mean_with_phase
    results = []

    # build initial grid
    grid = make_random_grid(seed_amplitude, seed_phase, p_dead, grid_size)

    # get live amplitudes and compute mean
    live_amplitudes = np.array([[cell[0] for cell in row] for row in grid], dtype=np.complex128)
    mean_abs = np.mean(np.abs(live_amplitudes))
    mean_with_phase = np.abs(np.mean(live_amplitudes))
    gen = 0

    # logic for rerunning if grid is dead
    dead_grid = 0
    rerun = False


    # store results
    results.append([gen, mean_abs, mean_with_phase])

    while gen < generations:
        grid = update_grid(grid)
        # Periodic measurement (skip gen==0)
        if measure_interval > 0 and gen > 0 and (gen % measure_interval == 0):
            grid = measurement(
                grid,
                rng_phase=rng_phase,
                rng_measurement=rng_measurement               
            )

        # get live amplitudes and compute mean
        live_amplitudes = np.array([[cell[0] for cell in row] for row in grid[:, :grid_size // 2]], dtype=np.complex128)
        mean_abs = np.mean(np.abs(live_amplitudes))
        mean_with_phase = np.abs(np.mean(live_amplitudes))
        gen += 1

        # store results
        results.append([gen, mean_abs, mean_with_phase])

        # breaks
        if mean_abs == 0:
            dead_grid += 1
        if dead_grid == 10:
            rerun = True
            break


    # save to csv
    csv_name = (
        f"results/data/"
        f"mRand_{random_measurement}_"
        f"pDead_{p_dead}_"
        f"mInt_{measure_interval}_"
        f"mDens_{measurement_density}_"
        f"ampSeed_{seed_amplitude}_"
        f"phaseSeed_{seed_phase}_"
        f"mSeed_{seed_measurement}_"
        f"gens_{gen}_"
        f"run_{run}"
        f".csv"
    )
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    np.savetxt(csv_name,
    np.array(results, dtype=float),
    delimiter=",",
    header="generation,mean_abs,mean_with_phase",
    comments=""
    )


    df = pd.read_csv(csv_name)

    # rename columns for plotting
    df = df.rename(columns={
        "mean_abs": "Mean |Amplitude|",
        "mean_with_phase": "|Mean Amplitude|"
    })

    # create plot
    df.plot(x="generation", y=["Mean |Amplitude|", "|Mean Amplitude|"])

    plt.xlabel("Generation")
    plt.ylabel("Mean Life")
    plt.title(f"Initial Dead: {p_dead} | Measure Interval: {measure_interval} | Measure Density: {measurement_density}")
    plt.tight_layout()

    # save plot
    plt.savefig(
    f"results/plots/"
    f"gridSize_{grid_size}_"
    f"mRandom_{random_measurement}_"
    f"pDead_{p_dead}_"
    f"mInterval_{measure_interval}_"
    f"mDensity_{measurement_density}_"
    f"ampSeed_{seed_amplitude}_"
    f"phaseSeed_{seed_phase}_"
    f"mSeed_{seed_measurement}_"
    f"gens_{gen}_"
    f"run_{run}"
    f".png", dpi=150)
    plt.close()

    return rerun


if __name__ == "__main__":
    p_deads = [0.8, 0.6, 0.4, 0.2, 0]
    measurement_intervals = [1, 3, 10, 32, 100]
    measurement_densities = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    random_measurements = [True, False]
    run_batch(p_deads, measurement_intervals, measurement_densities, random_measurements)


    # run_simulation(3555551, 11111, 11111,
    #                0.2, 0.4, 50,
    #                True, 1)
    
