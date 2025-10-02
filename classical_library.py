import numpy as np
import cmath
LIVE = np.array([1 + 0j, 0 + 0j])   # alive
DEAD = np.array([0 + 0j, 1 + 0j])   # dead

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
    (0, 4, np.array([LIVE.copy()[0] * cmath.exp(1j * cmath.pi), LIVE.copy()[1]]))],
    # 2x2 block

    # Beehive
    "beehive": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 2, LIVE.copy())
    ],

    # Loaf
    "loaf": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 2, LIVE.copy())
    ],

    # Boat
    "boat": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy())
    ],

    # Tub
    "tub": [
        (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy())
    ],

    # Ship (3x3)
    "ship": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 2, LIVE.copy())
    ],

    # Pond (4x4 ring of cells around hollow center)
    "pond": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy())
    ],

    # Carrier
    "carrier": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 1, LIVE.copy()), (2, 0, LIVE.copy()),
        (2, 3, LIVE.copy()), (3, 2, LIVE.copy()),
        (3, 3, LIVE.copy())
    ],

    # Snake
    "snake": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy())
    ],

    # Aircraft carrier (like 2 boats touching)
    "aircraft_carrier": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy())
    ],

    # Eater 1
    "eater1": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (2, 1, LIVE.copy()),
        (2, 3, LIVE.copy()), (3, 0, LIVE.copy()),
        (3, 1, LIVE.copy())
    ],

    # Eater 2
    "eater2": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (0, 2, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 2, LIVE.copy()),
        (3, 1, LIVE.copy())
    ],

    # Eater 3
    "eater3": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy()), (3, 0, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 3, LIVE.copy())
    ],

    # Eater 4
    "eater4": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (1, 0, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 3, LIVE.copy()), (3, 3, LIVE.copy()),
        (4, 0, LIVE.copy()), (4, 1, LIVE.copy())
    ],

    # Barge
    "barge": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 4, LIVE.copy()),
        (3, 2, LIVE.copy()), (3, 3, LIVE.copy())
    ],

    # Canoe
    "canoe": [
        (0, 1, LIVE.copy()), (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 3, LIVE.copy()), (3, 2, LIVE.copy())
    ],

    # Claw
    "claw": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy()), (4, 2, LIVE.copy())
    ],

    # Cis-shillelagh
    "cis_shillelagh": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (4, 2, LIVE.copy())
    ],

    # Long boat
    "long_boat": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 4, LIVE.copy()),
        (3, 2, LIVE.copy()), (3, 3, LIVE.copy()), (4, 3, LIVE.copy())
    ],

    # Pond with tail (sometimes called pi-heptomino-still)
    "pi_heptomino_still": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy()),
        (2, 4, LIVE.copy())
    ],

    # Toad (period 2)
    "toad": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), (0, 3, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), (1, 2, LIVE.copy())
    ],

    # Beacon (period 2)
    "beacon": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 2, LIVE.copy()), (3, 3, LIVE.copy())
    ],

    # Pulsar (period 3, large symmetric)
    "pulsar": [
        # Top left quadrant (others mirrored)
        (0, 2, LIVE.copy()), (0, 3, LIVE.copy()), (0, 4, LIVE.copy()),
        (0, 8, LIVE.copy()), (0, 9, LIVE.copy()), (0, 10, LIVE.copy()),
        (5, 2, LIVE.copy()), (5, 3, LIVE.copy()), (5, 4, LIVE.copy()),
        (5, 8, LIVE.copy()), (5, 9, LIVE.copy()), (5, 10, LIVE.copy()),
        (7, 2, LIVE.copy()), (7, 3, LIVE.copy()), (7, 4, LIVE.copy()),
        (7, 8, LIVE.copy()), (7, 9, LIVE.copy()), (7, 10, LIVE.copy()),
        (12, 2, LIVE.copy()), (12, 3, LIVE.copy()), (12, 4, LIVE.copy()),
        (12, 8, LIVE.copy()), (12, 9, LIVE.copy()), (12, 10, LIVE.copy()),
        (2, 0, LIVE.copy()), (3, 0, LIVE.copy()), (4, 0, LIVE.copy()),
        (8, 0, LIVE.copy()), (9, 0, LIVE.copy()), (10, 0, LIVE.copy()),
        (2, 5, LIVE.copy()), (3, 5, LIVE.copy()), (4, 5, LIVE.copy()),
        (8, 5, LIVE.copy()), (9, 5, LIVE.copy()), (10, 5, LIVE.copy()),
        (2, 7, LIVE.copy()), (3, 7, LIVE.copy()), (4, 7, LIVE.copy()),
        (8, 7, LIVE.copy()), (9, 7, LIVE.copy()), (10, 7, LIVE.copy()),
        (2, 12, LIVE.copy()), (3, 12, LIVE.copy()), (4, 12, LIVE.copy()),
        (8, 12, LIVE.copy()), (9, 12, LIVE.copy()), (10, 12, LIVE.copy())
    ],

    # Pentadecathlon (period 15)
    "pentadecathlon": [
        (0, 2, LIVE.copy()), (0, 3, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 5, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 5, LIVE.copy()),
        (3, 2, LIVE.copy()), (3, 3, LIVE.copy()),
        (4, 2, LIVE.copy()), (4, 3, LIVE.copy()),
        (5, 2, LIVE.copy()), (5, 3, LIVE.copy()),
        (6, 0, LIVE.copy()), (6, 5, LIVE.copy()),
        (7, 0, LIVE.copy()), (7, 5, LIVE.copy()),
        (8, 2, LIVE.copy()), (8, 3, LIVE.copy())
    ],

    # Clock (period 2 oscillator)
    "clock": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (1, 0, LIVE.copy()),
        (1, 2, LIVE.copy()), (2, 1, LIVE.copy()), (2, 2, LIVE.copy()),
        (3, 2, LIVE.copy()), (2, 3, LIVE.copy())
    ],

    # Unix (small period 3 oscillator, like bent line)
    "unix": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()),
        (1, 1, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 2, LIVE.copy()), (2, 3, LIVE.copy())
    ],

    # Cross (period 2, oscillates between + and x)
    "cross": [
        (0, 1, LIVE.copy()), (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 1, LIVE.copy())
    ],

    # Trans-blinker (blinker on a tail)
    "trans_blinker": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), (0, 3, LIVE.copy()),
        (1, 0, LIVE.copy()), (2, 0, LIVE.copy())
    ],

    # Mold (2 beehives exchanging)
    "mold": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 1, LIVE.copy()), (2, 2, LIVE.copy()),

        (0, 5, LIVE.copy()), (0, 6, LIVE.copy()),
        (1, 4, LIVE.copy()), (1, 7, LIVE.copy()),
        (2, 5, LIVE.copy()), (2, 6, LIVE.copy())
    ],

    # Trinity (3 blocks pattern, period 3)
    "trinity": [
        (0,0, LIVE.copy()), (0,1, LIVE.copy()), (1,0, LIVE.copy()), (1,1, LIVE.copy()),
        (3,0, LIVE.copy()), (3,1, LIVE.copy()), (4,0, LIVE.copy()), (4,1, LIVE.copy()),
        (6,0, LIVE.copy()), (6,1, LIVE.copy()), (7,0, LIVE.copy()), (7,1, LIVE.copy())
    ],

    # Two eaters (oscillator pair)
    "two_eaters": [
        (0,0, LIVE.copy()), (0,1, LIVE.copy()), (1,0, LIVE.copy()),
        (2,2, LIVE.copy()), (3,1, LIVE.copy()), (3,2, LIVE.copy())
    ],

    # Kok's galaxy (period 8)
    "kok_s_galaxy": [
        (0,0, LIVE.copy()), (0,1, LIVE.copy()), (1,0, LIVE.copy()), (1,1, LIVE.copy()),
        (0,4, LIVE.copy()), (0,5, LIVE.copy()), (1,4, LIVE.copy()), (1,5, LIVE.copy()),
        (4,0, LIVE.copy()), (4,1, LIVE.copy()), (5,0, LIVE.copy()), (5,1, LIVE.copy()),
        (4,4, LIVE.copy()), (4,5, LIVE.copy()), (5,4, LIVE.copy()), (5,5, LIVE.copy())
    ],

    # Baker's stamp (period 5)
    "baker_stamp": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), (0, 3, LIVE.copy()),
        (1, 0, LIVE.copy()), (2, 3, LIVE.copy()), (3, 1, LIVE.copy()), (3, 2, LIVE.copy()), (3, 3, LIVE.copy())
    ],

    # 2 pulsar quadrants
    "two_pulsar_quadrants": [
        (0,0,LIVE.copy()), (0,1,LIVE.copy()), (1,0,LIVE.copy()), (2,1,LIVE.copy())
    ],

    # Tumbler (period 14 oscillator)
    "p4_tumbler": [
        (0,0,LIVE.copy()), (0,1,LIVE.copy()), (0,2,LIVE.copy()),
        (2,0,LIVE.copy()), (2,1,LIVE.copy()), (2,2,LIVE.copy()),
        (4,0,LIVE.copy()), (4,1,LIVE.copy()), (4,2,LIVE.copy()),
        (5,1,LIVE.copy()), (6,0,LIVE.copy()), (6,2,LIVE.copy())
    ],

    # Caterer (p8 oscillator, also called "p8 ring")
    "p8_ring": [
        (0,1,LIVE.copy()), (0,2,LIVE.copy()), (1,0,LIVE.copy()), (1,3,LIVE.copy()),
        (2,1,LIVE.copy()), (2,2,LIVE.copy())
    ],

    # Laboratory (period 3 oscillator of blocks)
    "laboratory": [
        (0,0,LIVE.copy()), (0,1,LIVE.copy()), (1,0,LIVE.copy()), (1,1,LIVE.copy()),
        (0,3,LIVE.copy()), (0,4,LIVE.copy()), (1,3,LIVE.copy()), (1,4,LIVE.copy())
    ],

    # 24P10 (period 10 oscillator)
    "24P10": [
        (0,0,LIVE.copy()), (0,1,LIVE.copy()), (1,0,LIVE.copy()), (1,1,LIVE.copy()),
        (3,0,LIVE.copy()), (3,1,LIVE.copy()), (4,0,LIVE.copy()), (4,1,LIVE.copy())
    ],


    # LWSS: Lightweight spaceship (c/2 orthogonal)
    "lwss": [
        (0, 1, LIVE.copy()), (0, 4, LIVE.copy()),
        (1, 0, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 4, LIVE.copy()),
        (3, 0, LIVE.copy()), (3, 1, LIVE.copy()), (3, 2, LIVE.copy()), (3, 3, LIVE.copy()),
    ],

    # MWSS: Middleweight spaceship
    "mwss": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), (0, 3, LIVE.copy()), (0, 4, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 5, LIVE.copy()),
        (2, 5, LIVE.copy()),
        (3, 0, LIVE.copy()), (3, 4, LIVE.copy()), (3, 5, LIVE.copy()),
        (4, 1, LIVE.copy()), (4, 2, LIVE.copy()), (4, 3, LIVE.copy()), (4, 4, LIVE.copy()),
    ],

    # HWSS: Heavyweight spaceship
    "hwss": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), (0, 3, LIVE.copy()), (0, 4, LIVE.copy()), (0, 5, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 6, LIVE.copy()),
        (2, 6, LIVE.copy()),
        (3, 0, LIVE.copy()), (3, 5, LIVE.copy()), (3, 6, LIVE.copy()),
        (4, 1, LIVE.copy()), (4, 2, LIVE.copy()), (4, 3, LIVE.copy()), (4, 4, LIVE.copy()), (4, 5, LIVE.copy()),
    ],
    
    
    # B-heptomino (long-lived methuselah)
    "b_heptomino": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), (2, 1, LIVE.copy()), (2, 2, LIVE.copy())
    ],

    # Pi-heptomino
    "pi_heptomino": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 2, LIVE.copy())
    ],

    # Hen (small asymmetric methuselah)
    "hen": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 2, LIVE.copy())
    ],

    # Gosper Glider Gun (first infinite growth pattern)
    "gosper_glider_gun": [
        # Left square
        (5, 1, LIVE.copy()), (5, 2, LIVE.copy()),
        (6, 1, LIVE.copy()), (6, 2, LIVE.copy()),

        # Left-hand cluster
        (5, 11, LIVE.copy()), (6, 11, LIVE.copy()), (7, 11, LIVE.copy()),
        (4, 12, LIVE.copy()), (8, 12, LIVE.copy()),
        (3, 13, LIVE.copy()), (9, 13, LIVE.copy()),
        (3, 14, LIVE.copy()), (9, 14, LIVE.copy()),
        (6, 15, LIVE.copy()),
        (4, 16, LIVE.copy()), (8, 16, LIVE.copy()),
        (5, 17, LIVE.copy()), (6, 17, LIVE.copy()), (7, 17, LIVE.copy()),
        (6, 18, LIVE.copy()),

        # Right-hand cluster
        (3, 21, LIVE.copy()), (4, 21, LIVE.copy()), (5, 21, LIVE.copy()),
        (3, 22, LIVE.copy()), (4, 22, LIVE.copy()), (5, 22, LIVE.copy()),
        (2, 23, LIVE.copy()), (6, 23, LIVE.copy()),
        (1, 25, LIVE.copy()), (2, 25, LIVE.copy()), (6, 25, LIVE.copy()), (7, 25, LIVE.copy()),

        # Block on the far right
        (3, 35, LIVE.copy()), (4, 35, LIVE.copy()),
        (3, 36, LIVE.copy()), (4, 36, LIVE.copy()),
    ],

    # Schick Engine (classic small c/2 puffer, leaves debris behind)
    "schick_engine": [
        (0, 2, LIVE.copy()), (0, 3, LIVE.copy()), (0, 4, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), (1, 2, LIVE.copy()),
        (2, 2, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy()), (3, 3, LIVE.copy()),
    ],

    # Block-puffer (a simple puffer class that deposits blocks)
    # Small version: spaceship with trailing sparks that condense into blocks
    "block_puffer": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()),
        (1, 0, LIVE.copy()), (1, 3, LIVE.copy()),
        (2, 0, LIVE.copy()), (2, 3, LIVE.copy()),
        (3, 1, LIVE.copy()), (3, 2, LIVE.copy())
    ],

    # -- Glider Rake (minimal seed that drifts and sprays gliders) --
    "glider_rake": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), 
        (1, 1, LIVE.copy()), (1, 2, LIVE.copy()), 
        (2, 0, LIVE.copy()), (2, 2, LIVE.copy()), 
        (3, 3, LIVE.copy()), (4, 1, LIVE.copy()), (4, 2, LIVE.copy())
    ],

    # -- Pi Rake (uses Pi-heptomino as periodic glider source) --
    "pi_rake": [
        (0, 0, LIVE.copy()), (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), 
        (1, 0, LIVE.copy()), (1, 2, LIVE.copy()), 
        (2, 0, LIVE.copy()), (2, 2, LIVE.copy()), 
        (2, 4, LIVE.copy()), (3, 1, LIVE.copy()), (3, 3, LIVE.copy())
    ],

    # -- Switch Engine Rake (classic c/12 pattern) --
    "switch_engine_rake": [
        (0, 1, LIVE.copy()), (0, 2, LIVE.copy()), 
        (1, 0, LIVE.copy()), (1, 1, LIVE.copy()), 
        (1, 3, LIVE.copy()), 
        (2, 2, LIVE.copy()), 
        (3, 0, LIVE.copy()), (3, 2, LIVE.copy()), 
        (4, 1, LIVE.copy()), (4, 3, LIVE.copy()), (5, 2, LIVE.copy())
    ],
}