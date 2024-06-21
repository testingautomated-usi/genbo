####################
# DonkeyCar params #
####################

DONKEY_REFERENCE_TRACE_HEADER = "pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,rotation_angle,vel_x,vel_y,vel_z,forward_x,forward_y,forward_z"
DONKEY_REFERENCE_TRACE_HEADER_TYPES = "float,float,float,float,float,float,float,float,float,float,float,float,float,float"

DONKEY_REFERENCE_TRACE_USED_KEYS = (
    "pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,rotation_angle,vel_x,vel_z"
)

MAX_SPEED_DONKEY = 38  # needed to make the car go to 30 km/h max
# MAX_SPEED_DONKEY = 35  # needed to make the car go to 30 km/h max
# MAX_SPEED_DONKEY = 48  # needed to make the car go to 35 km/h max
# MAX_SPEED_DONKEY = 45
MIN_SPEED_DONKEY = 15

# 2 is the road width
EPS_POSITION_DONKEY_SANDBOX = 2 * 10 / 100
# 4 is the road width
# strict
EPS_POSITION_DONKEY_GENERATED = 4 * 10 / 100
EPS_VELOCITY_DONKEY = MAX_SPEED_DONKEY * 10 / 100
EPS_ORIENTATION_DONKEY = 360 * 2 / 100

# relaxed
# EPS_POSITION_DONKEY_GENERATED = 4 * 20 / 100
# EPS_VELOCITY_DONKEY = MAX_SPEED_DONKEY * 20 / 100
# EPS_ORIENTATION_DONKEY = 360 * 4 / 100

MAX_ORIENTATION_CHANGE_DONKEY = 20

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = -MAX_STEERING
MAX_STEERING_DIFF = 0.2

# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR_DONKEY = 3

BASE_PORT = 9091
BASE_SOCKET_LOCAL_ADDRESS = 52804

MAX_EPISODE_STEPS_DONKEY_SANDBOX = 250
MAX_EPISODE_STEPS_DONKEY_GENERATED_TRACK = 250

# PID constants SANDBOX_LAB
KP_SANDBOX_DONKEY = 1
KD_SANDBOX_DONKEY = 0.0
KI_SANDBOX_DONKEY = 0.0

# PID constants GENERATED_TRACK
KP_GENERATED_TRACK_DONKEY = 1.0
KD_GENERATED_TRACK_DONKEY = 0.0
KI_GENERATED_TRACK_DONKEY = 0.02
