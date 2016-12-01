# defines constants for the project.

BOARD_SZ = 9
PASS_ACTION = (BOARD_SZ**2)
RESIGN_ACTION = (BOARD_SZ**2)+1
KOMI = 6.5

# default feature list for the state generation
FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                "self_atari_size", "liberties_after", "ladder_capture", 
                "ladder_escape", "sensibleness", "zeros"]


#######################################################################
#																	  #
#			Supervised Learning Neural Network Constants			  #
#																	  #
#######################################################################

BATCH_SIZE = 32
NUM_LAYERS = 13
DEPTH = 48
NUM_ACTIONS = 81
NUM_TRAIN_SMALL = 16512
NUM_TRAIN_LARGE = 132288
CNN_REG_CONSTANTS = [0.6]*NUM_LAYERS
