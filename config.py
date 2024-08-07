DONKEY_SIM_NAME = "donkey"
MOCK_SIM_NAME = "mock"

REFERENCE_TRACE_FILENAME = "reference_trace.csv"
CURRENT_WAYPOINT_KEY = "current_waypoint"
REFERENCE_TRACE_CONSTANT_HEADER = "episode_counter,{}".format(CURRENT_WAYPOINT_KEY)
REFERENCE_TRACE_CONSTANT_HEADER_TYPES = "int,int"

SIMULATOR_NAMES = [DONKEY_SIM_NAME, MOCK_SIM_NAME]
AGENT_TYPE_RANDOM = "random"
AGENT_TYPE_SUPERVISED = "supervised"
AGENT_TYPE_AUTOPILOT = "autopilot"
AGENT_TYPES = [AGENT_TYPE_RANDOM, AGENT_TYPE_SUPERVISED, AGENT_TYPE_AUTOPILOT]
ROAD_TEST_GENERATOR_NAMES = ["constant"]

ROAD_WIDTH = 8.0
NUM_CONTROL_NODES = 8
NUM_SAMPLED_POINTS = 20
MAX_ANGLE = 270
MIN_ANGLE = 20
SEG_LENGTH = 25

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
