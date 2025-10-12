# examples/a3/spawns.py

Z_FLAT   = 0.30
Z_RUGGED = 0.30
Z_SLOPE  = 0.34
Z_FINISH = 0.36

SPAWNS = {
    # Section 1: Flat (x in roughly [-2.5, 0.5])
    "flat_start": (-2.20, 0.0, Z_FLAT),
    "flat_mid":   (-1.00, 0.0, Z_FLAT),
    "flat_end":   ( 0.30, 0.0, Z_FLAT),

    # Section 2: Rugged (x in roughly [0.5, 2.5])
    "rugged_start": (0.70, 0.0, Z_RUGGED),
    "rugged_mid":   (1.50, 0.0, Z_RUGGED),
    "rugged_end":   (2.30, 0.0, Z_RUGGED),

    # Section 3: Incline (x in roughly [2.48, 4.48])
    "slope_start": (2.60, 0.0, Z_SLOPE),
    "slope_mid":   (3.50, 0.0, Z_SLOPE),
    "slope_top":   (4.30, 0.0, Z_SLOPE),

    # Finish plateau (diagnostics)
    "finish": (5.20, 0.0, Z_FINISH),
}

def cycle_spawns(order=("flat_start","rugged_mid","slope_start","slope_mid","slope_top")):
    """Yield spawn tuples in a loop for multi-spawn training."""
    while True:
        for key in order:
            yield SPAWNS[key]
