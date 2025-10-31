""""
Script to perform multiple simulations over a range of variables.
Call run_batch(p_deads, measurement_intervals, measurement_densities, random_measurements),
where the first three are arrays, and the last one has set to be to [True] for now.

Simulations will be run until the target generation is met.
If the grid dies, a rerun will be done. For the last rerun, simulating will continue until target generation.

There are four plots created for every simulation:
 - plot_mean_amplitudes
    plots mean amplitudes; Mean |Amplitude| and |Mean Amplitude|

 - plot_life_probability
    plots probability of being alive, i.e. Mean |Amplitude|^2
 - plot_life_probability_with_phase
    adds |Mean Amplitude|^2 to plot_life_probability
 - plot_life_probability_with_amp_difference
    adds the amplitude difference of plot_mean_amplitudes to plot_life_probability
"""
from main import make_random_grid, update_grid, measurement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import itertools
import multiprocessing as mp


generations = 2500
max_reruns = 25
number_of_workers = None # If None, will use all-1
rerun_until_alive = False
    # if True only reruns parameter combination if it died,
    # if False reruns each combination max_reruns number of times

def check_directories():
    sub_directories = [
        "results/data/all",
        "results/data/finished",
        "results/plots/all/mean_amplitudes",
        "results/plots/finished/mean_amplitudes",
        "results/plots/all/life_probability",
        "results/plots/finished/life_probability",
        "results/plots/all/life_probability_with_phase",
        "results/plots/finished/life_probability_with_phase",
        "results/plots/all/life_probability_with_amp_difference",
        "results/plots/finished/life_probability_with_amp_difference"
    ]

    for sub in sub_directories:
        os.makedirs(sub, exist_ok=True)

def run_batch(tasks):

    # set number of workers for multiprocessing
    workers = number_of_workers if number_of_workers is not None else mp.cpu_count() - 1

    with mp.Pool(processes=workers) as pool:
        # Use pool.map to distribute the task list to run_single_task
        pool.map(run_single_task, tasks)

def run_single_task(task):
    grid_size = 50
    p_dead, measurement_interval, measurement_density, random_measurement = task
    run = 1
    while True:
        # generate new random seeds
        seed_amplitude = random.randint(1, 999999)
        seed_phase = random.randint(1, 999999)
        seed_measurement = random.randint(1, 999999)

        # seed_amplitude = 2
        # seed_phase = 2
        # seed_measurement = 2

        print(f"--- Run {run} ---")
        print(f"{p_dead = }, {measurement_interval = }, {measurement_density = }")
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

        if not rerun_until_alive:
            rerun  = run < max_reruns

        if rerun:
            run += 1
            print("Rerun requested — starting next run.\n")
            continue  # go back to the top (new seeds, next run)
        else:
            print("Simulation finished successfully.")
            break  # stop looping

def save_to_csv(csv_name, results):
    np.savetxt(csv_name,
    np.array(results, dtype=float),
    delimiter=",",
    header="generation,mean_abs,mean_abs_phase,mean_abs_diff,mean_alive,mean_alive_phase",
    comments=""
    )

# --------------- Plot Logic ---------------

def plot_mean_amplitudes(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun):
    df = pd.read_csv(csv_name)

    # rename columns for plotting
    df = df.rename(columns={
        "mean_abs": "Mean |Amplitude|",
        "mean_abs_phase": "|Mean Amplitude|"
    })

    # create plot
    df.plot(x="generation", y=["Mean |Amplitude|", "|Mean Amplitude|"])

    plt.xlabel("Generation")
    plt.ylabel("Mean Life Amplitude")
    plt.title(f"Initial Dead: {p_dead} | Measure Interval: {measure_interval} | Measure Density: {measurement_density}")
    plt.tight_layout()

    # save plot
    plt.savefig(
    f"results/plots/all/mean_amplitudes/"
    f"{simulation_name}"
    f".png", dpi=150)

    # filter and store plot if target generation is reached
    if not rerun:
        plt.savefig(
        f"results/plots/finished/mean_amplitudes/"
        f"{simulation_name}"
        f".png", dpi=150)
    
    plt.close()

def plot_life_probability(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun):
    df = pd.read_csv(csv_name)

    # rename columns for plotting
    df = df.rename(columns={
        "mean_alive": "Mean Probability Alive"
    })

    # create plot
    df.plot(x="generation", y=["Mean Probability Alive"])

    plt.xlabel("Generation")
    plt.ylabel("Mean Life Probability")
    plt.title(f"Initial Dead: {p_dead} | Measure Interval: {measure_interval} | Measure Density: {measurement_density}")
    plt.tight_layout()

    # save plot
    plt.savefig(
    f"results/plots/all/life_probability/"
    f"{simulation_name}"
    f".png", dpi=150)

    # filter and store plot if target generation is reached
    if not rerun:
        plt.savefig(
        f"results/plots/finished/life_probability/"
        f"{simulation_name}"
        f".png", dpi=150)
    
    plt.close()

def plot_life_probability_with_phase(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun):
    df = pd.read_csv(csv_name)

    # rename columns for plotting
    df = df.rename(columns={
        "mean_alive": "Mean Probability Alive",
        "mean_alive_phase": "|Mean Amplitude|^2"
    })

    # create plot
    df.plot(x="generation", y=["Mean Probability Alive", "|Mean Amplitude|^2"])

    plt.xlabel("Generation")
    plt.ylabel("Mean Life Probability")
    plt.title(f"Initial Dead: {p_dead} | Measure Interval: {measure_interval} | Measure Density: {measurement_density}")
    plt.tight_layout()

    # save plot
    plt.savefig(
    f"results/plots/all/life_probability_with_phase/"
    f"{simulation_name}"
    f".png", dpi=150)

    # filter and store plot if target generation is reached
    if not rerun:
        plt.savefig(
        f"results/plots/finished/life_probability_with_phase/"
        f"{simulation_name}"
        f".png", dpi=150)
    
    plt.close()

def plot_life_probability_with_amp_difference(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun):
    df = pd.read_csv(csv_name)

    # rename columns for plotting
    df = df.rename(columns={
        "mean_alive": "Mean Probability Alive",
        "mean_abs_diff": "Mean Amp Diff"
    })

    # create plot
    df.plot(x="generation", y=["Mean Probability Alive", "Mean Amp Diff"])

    plt.xlabel("Generation")
    plt.ylabel("Mean Life Probability")
    plt.title(f"Initial Dead: {p_dead} | Measure Interval: {measure_interval} | Measure Density: {measurement_density}")
    plt.tight_layout()

    # save plot
    plt.savefig(
    f"results/plots/all/life_probability_with_amp_difference/"
    f"{simulation_name}"
    f".png", dpi=150)

    # filter and store plot if target generation is reached
    if not rerun:
        plt.savefig(
        f"results/plots/finished/life_probability_with_amp_difference/"
        f"{simulation_name}"
        f".png", dpi=150)
    
    plt.close()

# -------------------------------------------

def run_simulation(seed_amplitude, seed_phase, seed_measurement,
                   p_dead, measurement_density, grid_size,
                   random_measurement, measure_interval, run):

    # build initial grid
    grid = make_random_grid(seed_amplitude, seed_phase, p_dead, grid_size)

    
    rng_phase = np.random.default_rng(seed_phase)
    rng_measurement = np.random.default_rng(seed_measurement)

    # get live amplitudes and compute mean
    live_amplitudes = np.array([[cell[0] for cell in row] for row in grid], dtype=np.complex128)
    mean_abs = np.mean(np.abs(live_amplitudes))
    mean_abs_phase = np.abs(np.mean(live_amplitudes))
    mean_abs_diff = mean_abs - mean_abs_phase

    mean_alive = np.mean(np.abs(live_amplitudes)**2)
    mean_alive_phase = np.abs(np.mean(live_amplitudes))**2
    gen = 0

    # logic for rerunning if grid is dead
    dead_grid = 0
    rerun = False

    # store results
    result = [gen, mean_abs, mean_abs_phase, mean_abs_diff, mean_alive, mean_alive_phase]
    results = np.zeros((generations+1, len(result)))  # not using dynamic length
    results[0] = result

    while gen < generations:
        grid = update_grid(grid)
        # Periodic measurement (skip gen==0)
        if measure_interval > 0 and gen > 0 and (gen % measure_interval == 0):
            grid = measurement(
                grid,
                rng_phase,
                rng_measurement,
                measurement_density,
                random_phase_upon_measurement= random_measurement
            )

        # get live amplitudes and compute mean
        live_amplitudes = np.array([[cell[0] for cell in row] for row in grid], dtype=np.complex128)
        mean_abs = np.mean(np.abs(live_amplitudes))
        mean_abs_phase = np.abs(np.mean(live_amplitudes))
        mean_abs_diff = mean_abs - mean_abs_phase


        mean_alive = np.mean(np.abs(live_amplitudes)**2)
        mean_alive_phase = np.abs(np.mean(live_amplitudes))**2
        gen += 1

        # store results, not using dynamic length
        result = [gen, mean_abs, mean_abs_phase, mean_abs_diff, mean_alive, mean_alive_phase]
        results[gen] = result

        # breaks
        if run < max_reruns and mean_abs == 0:
            rerun = True
            results = results[:gen+1]  # truncate length
            break


    simulation_name = (
        f"grid_{grid_size}_"
        f"pDead_{p_dead}_"
        f"interval_{measure_interval}_"
        f"density_{measurement_density}_"
        f"run_{run}_"
        f"gens_ {gen}_"
        f"ampSeed_{seed_amplitude}_"
        f"phaseSeed_{seed_phase}_"
        f"mSeed_{seed_measurement}"
    )

    # save to csv
    csv_name = (
        f"results/data/all/"
        f"{simulation_name}"
        f".csv"
    )
    save_to_csv(csv_name=csv_name, results=results)

    # filter for runs that reached target generation and also store seperately
    if not rerun:
        csv_name = (
        f"results/data/finished/"
        f"{simulation_name}"
        f".csv"
        )
        save_to_csv(csv_name=csv_name, results=results)


    # create plots with filtering, filter out what is not needed
    plot_mean_amplitudes(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun)
    plot_life_probability(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun)
    plot_life_probability_with_phase(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun)
    plot_life_probability_with_amp_difference(csv_name, simulation_name, p_dead, measure_interval, measurement_density, rerun)
    


    return rerun


if __name__ == "__main__":
    check_directories()

    n = 101
    p_deads = [0.4] * n
    measurement_intervals = [3] * n
    measurement_densities = np.linspace(0.3, 0.4, n)
    random_measurements = [True] * n

    # zip_longest
    tasks = list(itertools.zip_longest(p_deads, measurement_intervals, measurement_densities, random_measurements))

    run_batch(tasks)

