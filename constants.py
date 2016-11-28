# defines constants for the project.

BOARD_SZ = 9
PASS_ACTION = (BOARD_SZ**2)
RESIGN_ACTION = (BOARD_SZ**2)+1
KOMI = 6.5

# default feature list for the state generation
FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                "self_atari_size", "liberties_after", "ladder_capture", 
                "ladder_escape", "sensibleness", "zeros"]