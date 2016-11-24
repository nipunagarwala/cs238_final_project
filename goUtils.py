import gym
import pachi_py
import Rocgo
import numpy as np
import random
import utils

from Rocpreprocessing import Preprocess
from gym.envs.board_game import go

BOARD_SZ = 9

"""
pachi_py constants:
pachi_py.PASS_COORD     : -1 (corresponds to 81 after _coord_to_action())
pachi_py.RESIGN_COORD   : -2 (corresponds to 82 after _coord_to_action())
pachi_py.BLACK          : 1
pachi_py.WHITE          : 2
"""

def initGameStatus(boardSz=BOARD_SZ):
    # initialize the board state
    gs = Rocgo.GameState(boardSz)
    gs.current_player = Rocgo.BLACK

    return gs

class NNGoPlayer(object):

    # default feature list for the state generation
    FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                    "self_atari_size", "liberties_after", "ladder_capture", 
                    "ladder_escape", "sensibleness", "zeros"]
    BLACK = 1
    WHITE = 2

    def __init__(self, gymEnv, color, nnmodel, rocEnv=None, boardSz=BOARD_SZ):
        """
        @type   env         :   Go Environment Object
        @param  env         :   Go Environment
        @type   color       :   int
        @param  color       :   BLACK or WHITE
        """
        self.gymEnv = gymEnv
        self.color = color
        self.nnmodel = nnmodel
        if rocEnv:
            self.rocEnv = rocEnv
        else:
            self.rocEnv = initGameStatus()
        self.boardSz = boardSz

        self.pass_action = boardSz**2
        self.resign_action = boardSz**2+1
        self.states = []
        self.actions = []

    def get_legal_coords(self):
        """
        Returns legal moves
        @rtype              :   int array
        @rparam             :   Zero-indexed, row major coordinates of legal moves
        """
        coords = sorted([x+y*self.boardSz for (x,y) in self.rocEnv.get_legal_moves()])
        coords.append(self.pass_action)
        coords.append(self.resign_action)
        return coords

    def getState(self):
        """
        Returns the state information
        @rtype              :   numpy matrix
        @rparam             :   48x9x9 state info
        """
        nn_input = Preprocess(self.FEATURE_LIST).state_to_tensor(self.rocEnv)[0,:,:,:]
        return nn_input

    def intMove2rocMove(self, move):
        # converts integer move (0-82) to move coordinates used in Rochester board.
        if move==self.pass_action:
            return None
        else:
            return (move%self.boardSz,int(move/self.boardSz))

    def makemoveGym(self):
        """
        Plays a move against Pachi.
        Advances both the gym environment and the Rochester gameboard.
        NN always plays black for Gym (Pachi) games (thanks OpenAI...)

        Returns the reward
        """

        # check the state and store it
        state = self.getState()
        self.states.append(state)

        # move : Zero-indexed, row major coordinate to play
        # pass action is boardSz**2
        # resign action is boardSz**2+1

        # TODO: implement NN move decision logic here
        move = random.choice(self.get_legal_coords())
        print move

        # store the action chosen
        self.actions.append(move)

        # take the action in the OpenAI world
        observation,reward,done,info = self.gymEnv.step(move)

        if move==self.resign_action:
            return reward

        # update the Rochster board for my move
        self.rocEnv.do_move(self.intMove2rocMove(move))

        # update the Rochester board for the opponent's move
        if hasattr(self, 'last_obs'):
           oneLoc = np.nonzero(observation[NNGoPlayer.WHITE-1] - self.last_obs)
        else:
           oneLoc = np.nonzero(observation[NNGoPlayer.WHITE-1])
        
        if oneLoc:
            opponentMv = (oneLoc[1][0], oneLoc[0][0])
            self.rocEnv.do_move(opponentMv)
        else:
            self.rocEnv.do_move(None)

        self.last_obs = observation[1]

        return reward

    def makemoveRL(self):
        """
        Plays a move for RL learning.
        Only uses the Rochester gameboard.

        Returns True if the player resigns, False otherwise
        """
        
        # check the state and store it
        state = self.getState()
        self.states.append(state)

        # TODO: implement the move logic here
        move = random.choice(self.get_legal_coords())
        print move

        # store the action chosen
        self.actions.append(move)

        if move==self.resign_action:
            return True

        rocMove = self.intMove2rocMove(move)
        self.rocEnv.do_move(rocMove)
        return False

def postGameLearn(nnPlayer, reward):
    """
    NN backpropagation after a game is played out.
    """
    states = nnPlayer.states
    actions = nnPlayer.actions

    # learn, backprop
    # TODO

def Gym_Playout(policyModel):
    """
    Plays out a Gym game against Pachi.
    """
    gymEnv = gym.make('Go9x9-v0')
    gymEnv.reset()

    color = NNGoPlayer.BLACK

    nnPlayer = NNGoPlayer(gymEnv, color, policyModel)

    # play out the game
    while True:
        print '.'
        reward = nnPlayer.makemoveGym()
        if reward != 0:
            winner = Rocgo.BLACK if reward==1 else Rocgo.WHITE
            break

    print nnPlayer.getState()[0:3,:,:]
    nnPlayer.gymEnv.render()

    postGameLearn(nnPlayer,reward)
    print winner

Gym_Playout(None)

def RL_Playout(policyModel, opponentModel):
    """
    Plays out an RL game.
    """
    rocEnv = initGameStatus()

    policyColor = random.choice([Rocgo.BLACK, Rocgo.WHITE])

    # set the nnmodel consistent with policyColor
    blackModel = policyModel if policyColor==Rocgo.BLACK else opponentModel
    whiteModel = policyModel if policyColor==Rocgo.WHITE else opponentModel
    nnBlack = NNGoPlayer(None, NNGoPlayer.BLACK, blackModel, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(None, NNGoPlayer.WHITE, whiteModel, rocEnv=rocEnv)

    # play out the game
    while True:
        print '.'
        if nnBlack.makemoveRL():
            winner = Rocgo.WHITE
            break

        if nnWhite.makemoveRL():
            winner = Rocgo.BLACK
            break

        # did both players pass?
        if rocEnv.is_end_of_game:
            winner = rocEnv.get_winner()
            break

    print nnWhite.getState()[0:3,:,:]

    # end of the game tasks
    # determine who the winner is
    reward = 1 if (policyColor == winner) else -1

    # load the state-action pairs to learn from
    nnPlayer = nnBlack if policyColor==NNGoPlayer.BLACK else nnWhite

    postGameLearn(nnPlayer,reward)

    print ""
    print "Black Player" if policyColor==Rocgo.BLACK else "White Player"
    print reward
    print nnPlayer.actions