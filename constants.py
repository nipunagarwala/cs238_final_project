# defines constants for the project.

BOARD_SZ = 9
PASS_ACTION = (BOARD_SZ**2)
RESIGN_ACTION = (BOARD_SZ**2)+1
KOMI = 0

# default feature list for the state generation
FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                "self_atari_size", "liberties_after", "ladder_capture", 
                "ladder_escape", "sensibleness", "zeros"]


#######################################################################
#																	  #
#			Supervised Learning Neural Network Constants			  #
#																	  #
#######################################################################

BATCH_SIZE = 64
NUM_LAYERS = 12
DEPTH = 48
NUM_ACTIONS = 81
NUM_FILTERS = 192
NUM_FEATURES = 48
NUM_EPOCHS = 60
# NUM_TRAIN_SMALL = 16512
NUM_TRAIN_LARGE = 638912
# NUM_TRAIN_LARGE = 380160
# NUM_TRAIN_LARGE = 2280000
# NUM_TRAIN_LARGE = 220000

# NUM_TRAIN_LARGE = 26016
# NUM_TRAIN_LARGE = 48000
# CNN_REG_CONSTANTS = [0.9]*NUM_LAYERS
CNN_REG_CONSTANTS = None
RL_TRAIN_CHKPT = './rl_self_training-4'
# SL_CHECKPOINT
# RL_CHECKPOINT
# NUM_VAL_GAMES = 10000

