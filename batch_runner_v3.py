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


generations = 2500
max_reruns = 25

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
    # array to store gen,mean_abs, mean_with_phase
    results = []

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
    results.append([gen, mean_abs, mean_abs_phase, mean_abs_diff, mean_alive, mean_alive_phase])

    while gen < generations:
        grid = update_grid(grid)
        # Periodic measurement (skip gen==0)
        if measure_interval > 0 and gen > 0 and (gen % measure_interval == 0):
            grid = measurement(
                grid,
                rng_phase,
                rng_measurement,
                measurement_density
            )

        # get live amplitudes and compute mean
        live_amplitudes = np.array([[cell[0] for cell in row] for row in grid], dtype=np.complex128)
        mean_abs = np.mean(np.abs(live_amplitudes))
        mean_abs_phase = np.abs(np.mean(live_amplitudes))
        mean_abs_diff = mean_abs - mean_abs_phase


        mean_alive = np.mean(np.abs(live_amplitudes)**2)
        mean_alive_phase = np.abs(np.mean(live_amplitudes))**2
        gen += 1

        # store results
        results.append([gen, mean_abs, mean_abs_phase, mean_abs_diff, mean_alive, mean_alive_phase])

        # breaks
        if run < max_reruns and mean_abs == 0:
            rerun = True
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

    p_deads = [0.8, 0.6, 0.4, 0.2, 0]
    measurement_intervals = [1, 3, 10, 32, 100]
    measurement_densities = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    random_measurements = [True] # Don't change
    run_batch(p_deads, measurement_intervals, measurement_densities, random_measurements)


    # run_simulation(3555551, 11111, 11111,
    #                0.2, 0.4, 50,
    #                True, 1)
    
