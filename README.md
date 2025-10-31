# Quantum Game of Life

This project is an extension of an existing **semi-quantum version of Conway’s Game of Life**, implemented in Python with `pygame` for visualization and `numpy` for calculations.  
Each cell is represented as a quantum state with live/dead amplitudes, and evolution follows quantum rules with periodic measurements.

## Features
- Runs in **step-by-step mode** (press space to advance).
- Runs in **automatic mode** (updates on its own).
- Supports inserting classic patterns (blinker, block, glider, etc.) and quantum variations.
- Visualizes:
  - Probability of a cell being alive (grayscale shading).
  - Phase of the live amplitude (red arrows).

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/quantum-game-of-life.git
   cd quantum-game-of-life
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Simulation

Run the program with:

```bash
python main.py
```

You will be prompted to:
- Choose an **empty** or **random** starting grid.
- Optionally **insert patterns** at specific grid positions.
- Choose **step-by-step mode** (press space for each generation) or **automatic mode**.

Controls:
- **Spacebar** → Advance one generation (in step mode).  
- **Q** → Quit simulation.  
- **Window close button** → Exit.  

---

## Example Patterns
- **1** → Blinker  
- **2** → Block  
- **3** → Glider  
- **4** → Line  
- **5** → String  
- **6** → Phase Test  

You can insert multiple patterns before starting the simulation.


