import math
import pygame
import numpy as np
import cmath
import random
import copy
from classical_library import PATTERNS

#CELL_SIZE = 8       # Size of each cell in pixels for display
GRID_SIZE = 50          # Number of cells in each row/column of the grid
CELL_SIZE = max(1, int(800 // GRID_SIZE))  # Adjust CELL_SIZE to fit in 800x800 window
FPS = 5                 # Frames per second for automatic simulation
MEASURE_INTERVAL = 1   # Collapse quantum amplitudes every MEASURE_INTERVAL
MEASURE_DENSITY = 1  # Fraction of cells to measure during measurement
SEED_AMP = 12345
SEED_PHASE = 54321
SEED_MEASUREMENT = 98765
P_DEAD = 0.0              # Probability of a cell being DEAD in random grid
ONLY_DRAW_LIVE = True    # Only draw phase arrows for cells with live amplitude > 0
RANDOM_MEASUREMENT = False

""" Each cell is represented as a complex amplitude array: [live_amplitude, dead_amplitude]. """
LIVE = np.array([1 + 0j, 0 + 0j])  # Fully alive cell
DEAD = np.array([0 + 0j, 1 + 0j])  # Fully dead cell

def make_empty_grid(grid_size: int = GRID_SIZE):
    """
    Mathematica equivalent: MakeUni[n]
    Create an empty grid filled with dead cells. Each cell is a separate object to avoid shared references.

    Parameters:
        none.

    Returns:
        numpy.ndarray: GRID_SIZE x GRID_SIZE array where each element is a copy of DEAD cell.
    """
    grid = np.array([[DEAD.copy() for _ in range(grid_size)] for _ in range(grid_size)], dtype=object)
    return grid

def make_patterned_grid(pattern_name: str = None, x: int = None, y: int = None):

    """
    Create an empty grid of DEAD cells, 
    optionally seeded with a starting pattern from PATTERNS.

    Parameters:
        grid_size    : size of the square grid
        pattern_name : optional string key into PATTERNS
        x, y         : optional coordinates for placing the pattern (defaults to center)
    Returns:
        numpy.ndarray grid 
    """
    grid_size = GRID_SIZE
    grid = np.array([[DEAD.copy() for _ in range(grid_size)] for _ in range(grid_size)], dtype=object)
    
    if pattern_name and pattern_name in PATTERNS:
        # Default to center if no coordinates given
        if x is None: 
            x = grid_size // 2
        if y is None:
            y = grid_size // 2
        insert_pattern(grid, x, y, pattern_name)

    return grid


def random_cell(rng_amp, rng_phase, p_dead=P_DEAD):
    """
    Create a cell with seeded random live amplitude and random phase.

    Parameters:
        rng_amp   : np.random.Generator for amplitude
        rng_phase : np.random.Generator for phase
        p_dead    : float, probability of the cell being dead (0 to 1)

    Returns:
        numpy.ndarray: [live_amplitude*exp(i*phase), dead_amplitude]
    """

    if rng_amp.random() < p_dead:
        return DEAD.copy()

    # changed below to ensure uniform distribution of probability
    # live_amplitude = rng_amp.random()
    # dead_amplitude = np.sqrt(1 - live_amplitude ** 2)

    p_live = rng_amp.random()
    live_amplitude = np.sqrt(p_live)
    dead_amplitude = np.sqrt(1.0 - p_live)
    phase = rng_phase.uniform(0, 2 * np.pi)
    return np.array([live_amplitude * cmath.exp(1j * phase), dead_amplitude])

# TODO
# Before: make_random_grid(seed_amp = SEED_AMP, seed_phase = SEED_PHASE, p_dead=P_DEAD, grid_size: int = GRID_SIZE):
#
def make_random_grid(seed_amp = None, seed_phase = None, p_dead=P_DEAD, grid_size: int = GRID_SIZE):
    """
    Create a grid filled with random cells using two seeds.

    Parameters:
        seed_amp   : int, seed for amplitude RNG
        seed_phase : int, seed for phase RNG
        grid_size  : int, side length of the grid
        p_dead     : float, probability of the cell being dead (0 to 1)

    Returns:
        np.ndarray: grid_size x grid_size array of random cells
    """

    # TODO This should fix this function
    if seed_amp is None:
        seed_amp = SEED_AMP
    if seed_phase is None:
        seed_phase = SEED_PHASE


    rng_amp = np.random.default_rng(seed_amp)
    rng_phase = np.random.default_rng(seed_phase)

    grid = np.array(
        [[random_cell(rng_amp, rng_phase, p_dead) for _ in range(grid_size)]
         for _ in range(grid_size)],
        dtype=object
    )
    return grid

def normalise_cell(cell):
    """
    Mathematica equivalent: NormaliseCell[cell]
    Normalise the norm of a cell so the probability of being dead and alive summed is not larger than 1.

    Parameters:
        cell (numpy.ndarray): array [live_amplitude, dead_amplitude].

    Returns:
        numpy.ndarray: normalised cell. Default to dead cell.
    """
    norm = np.sqrt(abs(cell[0]) ** 2 + abs(cell[1]) ** 2)
    if norm > 0:
        return np.array([cell[0] / norm, cell[1] / norm])
    return DEAD.copy()

def count_neighbours(grid, x, y):
    """
    Mathematica equivalent: CountNeigh[univers, x, y]
    Sum live amplitudes of the 8 neighbouring cells using periodic boundary conditions.

    Parameters:
        grid (numpy.ndarray): current grid,
        x (int): row index of the target cell,
        y (int): column index of the target cell.

    Returns:
        complex: sum of live amplitudes of neighbours of cell at location (x, y).
    """
    N = len(grid)
    total = 0 + 0j
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # Cell itself is not a neighbour so skip
                continue
            # Computes neighbour coordinates, %n implements periodic boundary conditions
            nx = (x + dx) % N
            ny = (y + dy) % N
            total += grid[nx][ny][0]  # Complex sum of the live amplitudes of the neighbours
    return total

def birth(cell, phi):
    """
    Mathematica equivalent: B operator
    It takes part of the dead_amplitude and turns it into live_amplitude. The dead_amplitude contributes with a phase
    factor exp(i*phi) that comes from the neighbours (the phase captures how the complex amplitudes of neighbors
    interfere). After birth, the dead amplitude is set to zero.

    Parameters:
        cell (numpy.ndarray): current cell,
        phi (float): phase angle from neighbouring cells.

    Returns:
        numpy.ndarray: new cel after birth operation.
    """
    return np.array([cell[0] + cmath.exp(1j * phi) * abs(cell[1]), 0])

def death(cell, phi):
    """
    Mathematica equivalent: D operator
    It takes part of the alive_amplitude and turns it into dead_amplitude. The dead_amplitude contributes with a phase 
    factor exp(i*phi). After death, the live amplitude is set to zero.

    Parameters:
        cell (numpy.ndarray): current cell,
        phi (float): phase angle from neighbouring cells.

    Returns:
        numpy.ndarray: new cell after death operation.
    """
    return np.array([0, cell[1] + cmath.exp(1j * phi) * abs(cell[0])])

def compute_cell(grid, x, y):
    """
    Mathematica equivalent: Generation[universe, x, y]
    Compute the next state of a single cell using the semi-quantum life rules. Returns an updated normalized cell.

    Parameters:
        grid (numpy.ndarray): current grid,
        x (int): row index of the cell,
        y (int): column index of the cell.

    Returns:
        numpy.ndarray: normalized nest state of the cell.
    """
    cell = grid[x][y]
    neighbours = count_neighbours(grid, x, y)
    A = abs(neighbours)
    #((potential(x,y))*2*np.pi) unused, unsure if works
    phi = cmath.phase(neighbours) if neighbours !=0 else 0

    new_cell = np.array([0 + 0j, 0 + 0j]) # Initialize new cell as complex zero array
    # Apply the semi-quantum rules from the paper
    if A <=1 or A >= 4:
        new_cell = death(cell, phi)
    elif 1 < A <= 2:
        new_cell = (np.sqrt(2) + 1) * (A - 1) * cell + (2 - A) * death(cell, phi)
    elif 2 < A <= 3:
        new_cell = (np.sqrt(2) + 1) * (A - 2) * birth(cell, phi) + (3 - A) * cell
    elif 3 < A < 4:
        new_cell = (np.sqrt(2) + 1) * (A - 3) * death(cell, phi) + (4 - A) * birth(cell, phi)
    return normalise_cell(new_cell)

def potential(x,y,grid_size=GRID_SIZE):
    """
    Build a potential field over the grid in [0,1].
    
    Parameters:
        grid_size (int): number of grid points.
    Returns:
        np.ndarray: grid_size x grid_size array of potentials.
    """
    pot = (x + y) / (2*(grid_size-1))
    return pot

def update_grid(grid):
    """
    Mathematica equivalent: Next_generation[universe]
    Compute the next generation for the entire grid using the compute_cell function.
    TODO why not use copy.deepcopy(grid)?

    Parameters:
        grid (numpy.ndarray): current grid.

    Returns:
        numpy.ndarray: new grid after one generation.
    """
    
    N = len(grid)
    old_grid = [[cell.copy() for cell in row] for row in grid] # Deep copy to avoid in-pace changes
    new_grid = np.array([[compute_cell(old_grid, i, j) for j in range(N)] for i in range(N)],
                        dtype=object)
    return new_grid




            
            
# TODO the seeds need to be fixed

def measurement(grid, rng_phase, rng_measurement, measurement_density=MEASURE_DENSITY):
    """
    Collapses quantum probabilities on only a fraction of the grid cells.
    Parameters:
        grid (numpy.ndarray): current grid.
        density (float): fraction of cells to measure, between 0 and 1.
    Returns:
        numpy.ndarray: grid where each cell is either the same superposition
                       (if not measured) or collapsed to classical [1,0]/[0,1].
    """
    #rng_measurement = np.random.default_rng(measurement_seed)   # seeded RNG for determinism
    #rng_phase = np.random.default_rng(phase_seed)
    N = len(grid)
    new_grid = np.empty((N, N), dtype=object)

    for i in range(N):
        for j in range(N):
            random_measure = rng_measurement.random()   # random measure for deciding if measured
            if random_measure < measurement_density:
                # Perform measurement
                prob_alive = abs(grid[i][j][0]) ** 2
                collapse_random = rng_measurement.random()   # random measure deciding for collapse
                collapse_alive = (collapse_random < prob_alive)
                #reset phase to zero after measurement OR zero to 2pi random phase

                if collapse_alive:
                    if RANDOM_MEASUREMENT:
                        random_phase = cmath.exp(1j * rng_phase.uniform(0, 2*np.pi))
                        new_grid[i][j] = np.array([1 * random_phase, 0 + 0j])
                    else:
                        new_grid[i][j] = LIVE.copy()

                else: 
                    if RANDOM_MEASUREMENT:
                        random_phase = cmath.exp(1j * rng_phase.uniform(0, 2*np.pi))
                        new_grid[i][j] = np.array([0 + 0j, 1 * random_phase])
                    else:
                        new_grid[i][j] = DEAD.copy()
            else:
                # Leave unmeasured (carry over state)
                new_grid[i][j] = grid[i][j].copy()
    return new_grid
def insert_pattern(grid, x, y, pattern_name):
    """
    Insert a predefined pattern into the grid at a specific location.

    Parameters:
        grid (numpy.ndarray): grid to insert pattern into,
        x (int): row index to insert pattern.
        y (int): column index to insert pattern.

    Returns:
        numpy.ndarray: grid with the pattern inserted.
    """
    N = len(grid)
    for dx, dy, state in PATTERNS.get(pattern_name, []):
        px, py = (x + dx) % N, (y + dy) % N
        grid[px][py] = state.copy()
    return grid

def draw_arrow(screen, start, end, color=(0, 255, 0)):
    """
    Draw an arrow from start to end representing the phase on the screen.

    Parameters:
        screen: Pygame surface,
        start (tuple): (x, y) start coordinates,
        end (tuple): (x, y) end coordinates,
        color (tuple): RGB color.
    """
    pygame.draw.line(screen, color, start, end, 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = np.arctan2(dy, dx)
    size = 4
    # Computes two small lines (left and right arrowhead) rotated 30 degrees (to make tip look like an arrow)
    left = (end[0] - size * np.cos(angle - np.pi / 6), end[1] - size * np.sin(angle - np.pi / 6))
    right = (end[0] - size * np.cos(angle + np.pi / 6), end[1] - size * np.sin(angle + np.pi / 6))
    # Draw the two arrowhead lines.
    pygame.draw.line(screen, color, end, left, 2)
    pygame.draw.line(screen, color, end, right, 2)

def display_grid(screen, grid, only_draw_live=ONLY_DRAW_LIVE):
    """
    Display the current grid with grayscale representing alive probabilities and arrows showing the phase of the live
    coefficient.

    Parameters:
        screen (Pygame surface): the window or surface to draw on.
        grid (numpy.ndarray): current grid of cells, where each cell is an array [alive_amplitude, dead_amplitude]

    Returns:
        none. This function does not return a value but only updates the display.

    TODO check if calculations for drawing the arrows are correct
    Explanation:
        - Each cell is represented as a rectangle of size CELL_SIZE x CELL_SIZE.
        - The brightness of the rectangle corresponds to the probability of the cell being alive = |live_amplitude|^2
        - An arrow is drawn in each cell representing the phase of the alive amplitude. To draw the arrow in the correct
          direction use trigonometry:
          * cosine gives the horizontal movement (x-axis) from the center
          * sine gives the vertical movement (y-axis) from the center
          This converts the phase angle into a 2D arrow inside the cell. Pygame's y-axis increases downward, so to make
          the arrow point in the correct direction, we subtract the vertical component.
        - After drawing all cells and arrows, pygame.display.flip() updates the screen.

    Visual idea:
        Cell center is the starting point of the arrow The arrow tip is at (center_x + cos(phase)*length, center_y -
        sin(phase)*length) and the arrow points in the direction of the complex phase.
    """
    screen.fill((0, 0, 0))
    N = len(grid)
    arrow_length = CELL_SIZE * 0.3
    arrow_color = (255, 0, 0)

    for i in range(N):
        for j in range(N):
            cell = grid[i][j]

            # Draw background of the cell (grayscale based on alive probability)
            prob_alive = abs(cell[0]) ** 2
            color_value = int(prob_alive * 255)
            color = (color_value, ) * 3 # Shorthand for (x, x, x)
            rect = (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE) # Rectangular area that represent the cells
            pygame.draw.rect(screen, color, rect)

            # Draw arrow which represents the phase of the live amplitude
            cx, cy = j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2
            phase = cmath.phase(cell[0])
            #if only_draw_live and prob_alive > 0: # Only draw arrow if live amplitude > 0
            draw_arrow(screen, (cx, cy), (cx + arrow_length * np.cos(phase), cy - arrow_length *
                                        np.sin(phase)), arrow_color)
    pygame.display.flip()

def choose_grid():
    """
    Let user choose empty or random initial grid and optionally insert patterns.

    Returns:
         numpy.ndarray: initial grid with optional patterns.
    """

    choice = input("Choose 1 for empty grid, choose 2 for random grid, 3 for (pre-)patterned grid: ")
    def insert_manual_pattern():
        keys = {"1": "blinker", "2": "block", "3": "glider", "4": "line", "5": "string", "6": "phase_test"}
        while True:
            key = input("Choose pattern 1 (blinker), 2 (block), 3 (glider), 4 (line), 5 (string), 6 (phase test) or "
                        "'done': ")
            if key.lower() == "done":
                break
            if key not in keys:
                print("Invalid choice!")
                continue
            pattern = keys[key]
            x = int(input(f"Row to insert {pattern} (0-{GRID_SIZE - 1}): "))
            y = int(input(f"Column to insert {pattern} (0-{GRID_SIZE - 1}): "))
            insert_pattern(grid, x, y, pattern)
        
    def insert_pattern_library():
        while True:
            key = input("Type the pattern name (or 'done' to finish): ").strip().lower()
            if key == "done":
                break
            if key not in PATTERNS:
                print("Invalid pattern choice!")
                continue
            print(f"Selected pattern: {key}")
            return key

    # TODO change to match choice, somehow couldn't update python to make it work
    if choice == "1":
        grid = make_empty_grid()
        insert_manual_pattern()
    elif choice == "2":
        grid = make_random_grid()
        insert_manual_pattern()
    elif choice == "3":
        pattern_name = insert_pattern_library()
        grid = make_patterned_grid(pattern_name)
    else:
        print("Invalid choice")
        grid = make_empty_grid()

    return grid

def choose_mode():
    """
    Ask user whether to run step-by-step or automatic mode.

    Returns:
        bool: True if step-by-step, false if automatic.
    """
    choice = input("Choose 1 for step-by-step (space for next step), choose 2 for automatic: ")
    return choice == "1"

def main():
    """
    This is the main loop of the program, responsible for:

        - Initializing Pygame and creating the display window.
        - Asking the user to choose the initial grid and simulation mode.
        - Running the simulation loop, updating the grid each generation.
        - Handling user input for quitting (close button) or advancing generations (space) in step mode.
        - Periodically taking a measurement of the grid.
        - Updating the display after each generation.
    """
    pygame.init()

    # Choose initial grid and simulation mode
    grid = choose_grid()
    N = len(grid)

    
    screen = pygame.display.set_mode((N * CELL_SIZE, N * CELL_SIZE))
    clock = pygame.time.Clock()

    step_mode = choose_mode()
    rng_phase = np.random.default_rng(SEED_PHASE)
    rng_measurement = np.random.default_rng(SEED_MEASUREMENT)

    # Display the initial grid
    display_grid(screen, grid)

    running = True
    gen_count = 0

    while running:
        # Handle events (keyboard and quit button)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # User clicks close button, exit the  main loop
                running = False
            elif event.type == pygame.KEYDOWN:
                if step_mode and event.key == pygame.K_SPACE:
                    # User pressed space, advance one generation
                    grid = update_grid(grid)
                    gen_count += 1
                    display_grid(screen, grid)
                elif event.key == pygame.K_q: # Quit if user presses Q
                    running = False

        # Automatic mode, advance automatically without waiting for user input
        if not step_mode:
            grid = update_grid(grid)
            gen_count += 1
            display_grid(screen, grid)

        # Periodic measurement
        if gen_count % MEASURE_INTERVAL == 0 and gen_count > 0:
            grid = measurement(grid, rng_phase, rng_measurement)

        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()
