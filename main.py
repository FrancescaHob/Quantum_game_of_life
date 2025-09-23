import pygame
import numpy as np
import cmath
import random
import copy

CELL_SIZE = 30          # Size of each cell in pixels for display
GRID_SIZE = 20          # Number of cells in each row/column of the grid
FPS = 5                 # Frames per second for automatic simulation
MEASURE_INTERVAL = 10   # Collapse quantum amplitudes every MEASURE_INTERVAL

""" Each cell is represented as a complex amplitude array: [live_amplitude, dead_amplitude]. """
LIVE = np.array([1 + 0j, 0 + 0j])  # Fully alive cell
DEAD = np.array([0 + 0j, 1 + 0j])  # Fully dead cell

""" Pattern dictionary which we use in combination with insert_pattern(grid, x, y, pattern_name). """
PATTERNS = {
    "blinker": [(0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (0, 2, LIVE.copy())],
    "block": [(0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (1, 0, LIVE.copy()), (1, 1, LIVE.copy())],
    "glider": [(0, 1, LIVE.copy()), (-1, 2, LIVE.copy()), (-2, 0, LIVE.copy()), (-2, 1, LIVE.copy()),
               (-2, 2, LIVE.copy())],
    "line": [(0, i, LIVE.copy()) for i in range(5)],
    "string": [(0, i, np.array([(-1) ** i, 0])) for i in range(5)],
    "phase_test": [(0, 0, np.array([LIVE.copy()[0] * cmath.exp(1j * 0), LIVE.copy()[1]])),
    (0, 1, np.array([LIVE.copy()[0] * cmath.exp(1j * cmath.pi / 4), LIVE.copy()[1]])),
    (0, 2, np.array([LIVE.copy()[0] * cmath.exp(1j * cmath.pi / 2), LIVE.copy()[1]])),
    (0, 3, np.array([LIVE.copy()[0] * cmath.exp(1j * 3 * cmath.pi / 4), LIVE.copy()[1]])),
    (0, 4, np.array([LIVE.copy()[0] * cmath.exp(1j * cmath.pi), LIVE.copy()[1]]))]
}

def make_empty_grid():
    """
    Mathematica equivalent: MakeUni[n]
    Create an empty grid filled with dead cells. Each cell is a separate object to avoid shared references.

    Parameters:
        none.

    Returns:
        numpy.ndarray: GRID_SIZE x GRID_SIZE array where each element is a copy of DEAD cell.
    """
    grid = np.array([[DEAD.copy() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)], dtype=object)
    return grid

def random_cell():
    """
    Create a cell with random live amplitude and random phase.
    
    Parameters: 
        none. 
        
    Returns: 
        numpy.ndarray: array [live_amplitude, dead_amplitude] with random live phase. 
    """
    live_amplitude = random.random()
    dead_amplitude = np.sqrt(1 - live_amplitude ** 2)
    phase = random.uniform(0, 2 * np.pi)
    return np.array([live_amplitude * cmath.exp(1j * phase), dead_amplitude])

def make_random_grid():
    """
    Create a grid filled with random cells.

    Parameters:
        none.

    Returns:
        numpy.ndarray: GRID_SIZE x GRID_SIZE array where each element is a random cell.
    """
    grid = np.array([[random_cell().copy() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)], dtype=object)
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
    total = 0 + 0j
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # Cell itself is not a neighbour so skip
                continue
            # Computes neighbour coordinates, %n implements periodic boundary conditions
            nx = (x + dx) % GRID_SIZE
            ny = (y + dy) % GRID_SIZE
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
    old_grid = [[cell.copy() for cell in row] for row in grid] # Deep copy to avoid in-pace changes
    new_grid = np.array([[compute_cell(old_grid, i, j) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)],
                        dtype=object)
    return new_grid

def measurement(grid, measurement_density=0.0):
    """
    Collapses quantum probabilities on only a fraction of the grid cells.
    Parameters:
        grid (numpy.ndarray): current grid.
        density (float): fraction of cells to measure, between 0 and 1.
    Returns:
        numpy.ndarray: grid where each cell is either the same superposition
                       (if not measured) or collapsed to classical [1,0]/[0,1].
    """
    new_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if random.random() < measurement_density:
                # Perform measurement
                prob_alive = abs(grid[i][j][0]) ** 2
                print("Hoi!",grid[i][j])
                new_grid[i][j] = np.array(LIVE.copy()*cmath.phase(grid[i][j][0])) if random.random() < prob_alive else (DEAD.copy()*cmath.phase(grid[i][j][1]))
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
    for dx, dy, state in PATTERNS.get(pattern_name, []):
        px, py = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
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

def display_grid(screen, grid):
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
    arrow_length = CELL_SIZE * 0.3
    arrow_color = (255, 0, 0)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
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
            draw_arrow(screen, (cx, cy), (cx + arrow_length * np.cos(phase), cy - arrow_length *
                                          np.sin(phase)), arrow_color)
    pygame.display.flip()

def choose_grid():
    """
    Let user choose empty or random initial grid and optionally insert patterns.

    Returns:
         numpy.ndarray: initial grid with optional patterns.
    """
    choice = input("Choose 1 for empty grid, choose 2 for random grid: ")
    grid = make_empty_grid() if choice == "1" else make_random_grid()
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
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    clock = pygame.time.Clock()

    # Choose initial grid and simulation mode
    grid = choose_grid()
    step_mode = choose_mode()

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
            grid = measurement(grid)

        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()