import math

# Air3D params (pursuit-evasion, 3D)
BETA = 0.25
VELOCITY = 0.75
OMEGA_MAX = 3.0
T_MAX = 1.0
T_STEP = 0.05

# Air3D grid bounds
X1_BOUNDS = (-1.0, 1.0)
X2_BOUNDS = (-1.0, 1.0)
X3_BOUNDS = (-math.pi, math.pi)

# Dubins3D params (goal reaching, 3D)
GOAL_R = 0.25
DUBINS_VELOCITY = 0.6
DUBINS_OMEGA_MAX = 1.1
DUBINS_ANGLE_ALPHA = 1.2
DUBINS_SET_MODE = "avoid"

# Dubins3D grid bounds
D_X_BOUNDS = (-1.0, 1.0)
D_Y_BOUNDS = (-1.0, 1.0)
D_THETA_BOUNDS = (-math.pi, math.pi)

# Grid resolution
GRID_POINTS = 101
