import gym
import pachi_py
import Rocgo
import numpy as np
import random

from constants import *
from rochesterWrappers import *
from gym.envs.board_game import go

"""
pachi_py constants:
pachi_py.PASS_COORD     : -1 (corresponds to 81 after _coord_to_action())
pachi_py.RESIGN_COORD   : -2 (corresponds to 82 after _coord_to_action())
pachi_py.BLACK          : 1
pachi_py.WHITE          : 2
"""

def nn_vs_nnGame(rocEnv, playBlack, nnBlack, nnWhite, verbose=False, playbyplay=False):
    # play out the game, returns the winner (NNGoPlayer.BLACK or NNGoPlayer.WHITE)
    counter = 0
    while True:
        if playbyplay:
            printRocBoard(nnBlack.rocEnv)

        nnPlayer = nnBlack if playBlack else nnWhite
        if nnPlayer.makemoveRL(playbyplay=playbyplay):
            winner = NNGoPlayer.WHITE if playBlack else NNGoPlayer.BLACK
            break
        playBlack = not playBlack

        # did both players pass?
        if counter>MAX_MOVES_PER_GAME or rocEnv.is_end_of_game:
            won = rocEnv.get_winner(verbose=verbose)
            if won == 0: # tie
                return None
            winner = NNGoPlayer.BLACK if won==Rocgo.BLACK else NNGoPlayer.WHITE
            break

        counter += 1

    return winner

class NNGoPlayer(object):
    """
    Implements the Neural Net powered Go player.
    """

    BLACK = 0
    WHITE = 1

    def __init__(self, color, nnmodel, gymEnv=None, rocEnv=None):
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
            self.rocEnv = initRocBoard()

        self.rocColor = Rocgo.BLACK if color==NNGoPlayer.BLACK else Rocgo.WHITE
        self.pachiColor = pachi_py.BLACK if color==NNGoPlayer.BLACK else pachi_py.WHITE
        self.states = []
        self.actions = []

    def makeRandomValidMove(self):
        return random.choice(get_legal_coords(self.rocEnv))

    def nnMoveLogic(self, state):
        # TODO: implement NN move decision logic here

        # move : Zero-indexed, row major coordinate to play
        # pass action is PASS_ACTION
        # resign action is RESIGN_ACTION
        incDimState = np.zeros((1, BOARD_SZ,BOARD_SZ, NUM_FEATURES))
        transStates = np.transpose(state, axes=[1,2,0])
        incDimState[0,:,:,:] = transStates


        pyx = (self.nnmodel).make_move(incDimState)
        predSortIndex = np.argsort(pyx)
        legal_actions = get_legal_coords(self.rocEnv)
        pyx = pyx[0][0]

        for action in predSortIndex[0][0]:
            if action in legal_actions:
                continue
            else:
                pyx[action] = 0

        pyx = [float(i)/sum(pyx) for i in pyx]

        actProb = random.uniform(0, 1)

        for action in range(0,NUM_ACTIONS):
            actProb -= pyx[action]
            if actProb <= 0:
                return action

        return PASS_ACTION

        # return self.makeRandomValidMove()

    def updatePachiMove(self, observation, playbyplay=False):
        """
        Infer the pachi movement from observations.
        """
        pachiBoard = observation[(self.color+1)%2]

        if hasattr(self, 'last_obs'):
            oneLoc = np.where((pachiBoard - self.last_obs)>0)
        else:
            oneLoc = np.where(pachiBoard>0)

        pachiRocColor = Rocgo.WHITE if self.color==NNGoPlayer.BLACK else Rocgo.BLACK
        if oneLoc[0].size==0:
            opponentMv = None
            self.last_pachi_mv = PASS_ACTION
        else:
            opponentMv = (oneLoc[1][0], oneLoc[0][0])
            self.last_pachi_mv = opponentMv[0]+opponentMv[1]*BOARD_SZ
        
        self.rocEnv.do_move(opponentMv, color=pachiRocColor)
        if playbyplay:
            if opponentMv:
                print "Pachi (%s) - %d" % ("Black" if self.color==NNGoPlayer.WHITE else "White",
                                           opponentMv[0]+opponentMv[1]*BOARD_SZ)
            else:
                print "Pachi (%s) - %s" % ("Black" if self.color==NNGoPlayer.WHITE else "White", opponentMv)

        self.last_obs = pachiBoard


    def makemoveGym(self, move=-1, playbyplay=False):
        """
        Plays a move against Pachi.
        Advances both the gym environment and the Rochester gameboard.

        Returns True if the game ended, False otherwise
        """

        # check the state and store it
        state = rocBoard2State(self.rocEnv)
        self.states.append(state)

        if move==-1:
            move = self.nnMoveLogic(state)
        if playbyplay:
            print "Policy (%s) - %d" % ("Black" if self.color==NNGoPlayer.BLACK else "White", move)

        # store the action chosen
        self.actions.append(move)

        # take the action in the OpenAI world
        observation,reward,_,_ = self.gymEnv.step(move)

        if move==RESIGN_ACTION:
            return True

        # update the Rochster board for my move
        self.rocEnv.do_move(intMove2rocMove(move), color=self.rocColor)

        # update the Rochester board for the opponent's move
        self.updatePachiMove(observation, playbyplay=playbyplay)

        return self.gymEnv.state.board.is_terminal
        
    def makemoveRL(self, playRandom=False, playbyplay=False):
        """
        Plays a move for RL learning.
        Only uses the Rochester gameboard.

        Returns True if the player resigns, False otherwise
        """
        
        # check the state and store it
        state = rocBoard2State(self.rocEnv)
        self.states.append(state)

        if playRandom:
            move = self.makeRandomValidMove()
        else:
            move = self.nnMoveLogic(state)

        if playbyplay:
            print "%s - move: %d" %("Black" if self.color==NNGoPlayer.BLACK else "White", move)

        # store the action chosen
        self.actions.append(move)

        if move==RESIGN_ACTION:
            return True

        rocMove = intMove2rocMove(move)
        self.rocEnv.do_move(rocMove, color=self.rocColor)
        return False
