import Rocgo
import numpy as np

from constants import *
from Rocpreprocessing import Preprocess

def initRocBoard():
    # initialize the board state
    gs = Rocgo.GameState(BOARD_SZ)
    gs.current_player = Rocgo.BLACK
    gs.komi = KOMI

    return gs

def rocBoard2State(rocEnv):
    """
    Returns the state information
    @rtype              :   numpy matrix
    @rparam             :   48x9x9 state info
    """
    nn_input = Preprocess(FEATURE_LIST).state_to_tensor(rocEnv)[0,:,:,:]
    return nn_input

def printRocBoard(rocEnv):
    one = np.transpose(Preprocess(FEATURE_LIST).state_to_tensor(rocEnv)[0,0,:,:])
    two = np.transpose(Preprocess(FEATURE_LIST).state_to_tensor(rocEnv)[0,1,:,:])
    print "Rochester Board"
    print "0=empty, 1=black, 2=white"
    oneModifier = 1 if rocEnv.current_player==Rocgo.BLACK else 2
    twoModifier = 2 if rocEnv.current_player==Rocgo.BLACK else 1
    print one*oneModifier + two*twoModifier

def returnRocBoard(rocEnv):
    one = np.transpose(Preprocess(FEATURE_LIST).state_to_tensor(rocEnv)[0,0,:,:])
    two = np.transpose(Preprocess(FEATURE_LIST).state_to_tensor(rocEnv)[0,1,:,:])
    oneModifier = 1 if rocEnv.current_player==Rocgo.BLACK else 2
    twoModifier = 2 if rocEnv.current_player==Rocgo.BLACK else 1
    return one*oneModifier + two*twoModifier

def get_legal_coords(rocEnv):
    """
    Returns legal moves
    @rtype              :   int array
    @rparam             :   Zero-indexed, row major coordinates of legal moves
    """
    coords = sorted([x+y*BOARD_SZ for (x,y) in rocEnv.get_legal_moves()])
    coords.append(PASS_ACTION)
    coords.append(RESIGN_ACTION)
    return coords

def intMove2rocMove(move):
    # converts integer move (0-82) to move coordinates used in Rochester board.
    if move==PASS_ACTION:
        return None
    else:
        return (move%BOARD_SZ,int(move/BOARD_SZ))