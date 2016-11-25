import gym
import pachi_py
import Rocgo
import numpy as np
import random
import utils

from Rocpreprocessing import Preprocess
from gym.envs.board_game import go

BOARD_SZ = 9
KOMI = 6.5

"""
pachi_py constants:
pachi_py.PASS_COORD     : -1 (corresponds to 81 after _coord_to_action())
pachi_py.RESIGN_COORD   : -2 (corresponds to 82 after _coord_to_action())
pachi_py.BLACK          : 1
pachi_py.WHITE          : 2
"""

class NNGoPlayer(object):
    """
    Implements the Neural Net powered Go player.
    """

    # default feature list for the state generation
    FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                    "self_atari_size", "liberties_after", "ladder_capture", 
                    "ladder_escape", "sensibleness", "zeros"]
    BLACK = 0
    WHITE = 1

    def __init__(self, color, nnmodel, gymEnv=None, rocEnv=None, boardSz=BOARD_SZ):
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
        self.boardSz = boardSz

        self.rocColor = Rocgo.BLACK if color==NNGoPlayer.BLACK else Rocgo.WHITE
        self.pachiColor = pachi_py.BLACK if color==NNGoPlayer.BLACK else pachi_py.WHITE
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

    def printRocBoard(self):
        one = np.transpose(Preprocess(self.FEATURE_LIST).state_to_tensor(self.rocEnv)[0,0,:,:])
        two = np.transpose(Preprocess(self.FEATURE_LIST).state_to_tensor(self.rocEnv)[0,1,:,:])
        print "Rochester Board"
        print "0=empty, 1=black, 2=white"
        oneModifier = 1 if self.rocEnv.current_player==Rocgo.BLACK else 2
        twoModifier = 2 if self.rocEnv.current_player==Rocgo.BLACK else 1
        print one*oneModifier + two*twoModifier

    def intMove2rocMove(self, move):
        # converts integer move (0-82) to move coordinates used in Rochester board.
        if move==self.pass_action:
            return None
        else:
            return (move%self.boardSz,int(move/self.boardSz))

    def makeRandomValidMove(self):
        return random.choice(self.get_legal_coords())

    def nnMoveLogic(self, state):
        # TODO: implement NN move decision logic here

        # move : Zero-indexed, row major coordinate to play
        # pass action is self.pass_action
        # resign action is self.resign_action

        return self.makeRandomValidMove()

    def updatePachiMove(self, observation, playbyplay=False):
        pachiBoard = observation[(self.color+1)%2]

        if hasattr(self, 'last_obs'):
            oneLoc = np.where((pachiBoard - self.last_obs)>0)
        else:
            oneLoc = np.where(pachiBoard>0)

        pachiRocColor = Rocgo.WHITE if self.color==NNGoPlayer.BLACK else Rocgo.BLACK
        if oneLoc[0].size==0:
            opponentMv = None
        else:
            opponentMv = (oneLoc[1][0], oneLoc[0][0])
        
        self.rocEnv.do_move(opponentMv, color=pachiRocColor)
        if playbyplay:
            if opponentMv:
                print "Pachi (%s) - %d" % ("Black" if self.color==NNGoPlayer.WHITE else "White",
                                           opponentMv[0]+opponentMv[1]*self.boardSz)
            else:
                print "Pachi (%s) - %s" % ("Black" if self.color==NNGoPlayer.WHITE else "White", opponentMv)

        self.last_obs = pachiBoard

    def makemoveGym(self, playbyplay=False):
        """
        Plays a move against Pachi.
        Advances both the gym environment and the Rochester gameboard.

        Returns True if the game ended, False otherwise
        """

        # check the state and store it
        state = self.getState()
        self.states.append(state)

        move = self.nnMoveLogic(state)
        if playbyplay:
            print "Policy (%s) - %d" % ("Black" if self.color==NNGoPlayer.BLACK else "White", move)

        # store the action chosen
        self.actions.append(move)

        # take the action in the OpenAI world
        observation,reward,_,_ = self.gymEnv.step(move)

        if move==self.resign_action:
            return True

        # update the Rochster board for my move
        self.rocEnv.do_move(self.intMove2rocMove(move), color=self.rocColor)

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
        state = self.getState()
        self.states.append(state)

        if playRandom:
            move = self.makeRandomValidMove()
        else:
            move = self.nnMoveLogic(state)

        if playbyplay:
            print "%s - move: %d" %("Black" if self.color==NNGoPlayer.BLACK else "White", move)

        # store the action chosen
        self.actions.append(move)

        if move==self.resign_action:
            return True

        rocMove = self.intMove2rocMove(move)
        self.rocEnv.do_move(rocMove, color=self.rocColor)
        return False

def initRocBoard(boardSz=BOARD_SZ):
    # initialize the board state
    gs = Rocgo.GameState(boardSz)
    gs.current_player = Rocgo.BLACK
    gs.komi = KOMI

    return gs

def postGameLearn(nnPlayer, reward):
    """
    NN backpropagation after a game is played out.
    """
    states = nnPlayer.states
    actions = nnPlayer.actions

    # learn, backprop
    # TODO

def Gym_Playout(policyModel, verbose=False, playbyplay=False):
    """
    Plays out a Gym game against Pachi.

    Return True if a game played had a winner. False if the game is a tie.
    """
    gymEnv = gym.make('Go9x9-v0')
    gymEnv.reset()

    # randomly choose a side to play
    policyColor = random.choice([NNGoPlayer.BLACK, NNGoPlayer.WHITE])
    nnPlayer = NNGoPlayer(policyColor, policyModel, gymEnv=gymEnv)
    if verbose:
        print "Playing Black" if policyColor==NNGoPlayer.BLACK else "Playing White"

    gymEnv.state.color = nnPlayer.pachiColor
    gymEnv.player_color = nnPlayer.pachiColor

    # if playing white, pass the first move to let Pachi play first
    if policyColor==NNGoPlayer.WHITE:
        observation,_,_,_ = gymEnv.step(nnPlayer.pass_action)
        nnPlayer.updatePachiMove(observation, playbyplay=playbyplay)

    # play out the game
    while True:
        if playbyplay:
            gymEnv.render()
            nnPlayer.printRocBoard()

        if nnPlayer.makemoveGym(playbyplay=playbyplay):
            break

    if gymEnv.state.board.official_score==0:
        return False

    winner = NNGoPlayer.BLACK if gymEnv.state.board.official_score<0 else NNGoPlayer.WHITE
    reward = 1 if (policyColor == winner) else -1
    
    postGameLearn(nnPlayer,reward)

    if verbose:
        nnPlayer.gymEnv.render()
        nnPlayer.printRocBoard()
        print ""
        print "Score: %d" % gymEnv.state.board.official_score
        print "Did I win? %d" % reward
        print "Actions taken: %s" %str(nnPlayer.actions)

    return True

def RL_Playout(policyModel, opponentModel, verbose=False, playbyplay=False):
    """
    Plays out an RL game.

    Return True if a game played had a winner. False if the game is a tie.
    """
    rocEnv = initRocBoard()

    policyColor = random.choice([NNGoPlayer.BLACK, NNGoPlayer.WHITE])
    if verbose:
        print "Playing Black" if policyColor==NNGoPlayer.BLACK else "Playing White"

    # set the nnmodel consistent with policyColor
    blackModel = policyModel if policyColor==Rocgo.BLACK else opponentModel
    whiteModel = policyModel if policyColor==Rocgo.WHITE else opponentModel
    nnBlack = NNGoPlayer(NNGoPlayer.BLACK, blackModel, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(NNGoPlayer.WHITE, whiteModel, rocEnv=rocEnv)

    # play out the game
    playBlack = True
    while True:
        if playbyplay:
            nnBlack.printRocBoard()

        nnPlayer = nnBlack if playBlack else nnWhite
        playBlack = not playBlack
        if nnPlayer.makemoveRL(playbyplay=playbyplay):
            winner = NNGoPlayer.WHITE if playBlack else NNGoPlayer.BLACK
            break

        # did both players pass?
        if rocEnv.is_end_of_game:
            won = rocEnv.get_winner(verbose=verbose)
            if won == 0: # tie
                return False
            winner = NNGoPlayer.BLACK if won==Rocgo.BLACK else NNGoPlayer.WHITE
            break

    # end of the game tasks
    # determine who the winner is
    reward = 1 if (policyColor == winner) else -1

    # load the state-action pairs to learn from
    nnPlayer = nnBlack if policyColor==NNGoPlayer.BLACK else nnWhite

    postGameLearn(nnPlayer,reward)

    if verbose:
        nnPlayer.printRocBoard()
        print ""
        print "Winner: %s" %("Black Player" if winner==NNGoPlayer.BLACK else "White Player")
        print "Did I win? %d" % reward
        print "Actions taken: %s" %str(nnPlayer.actions)

    return True

def valueDataGen(sl_model, rl_model, U_MAX=90, verbose=False, playbyplay=False):
    """
    Generates state to win/loss pair for the use in value network learning

    Returns (state at time U),(reward)
    Returns nothing if a game played was a tie
    """

    # SL phase- play out the SL policies to U-1 timestep
    blackPlays = True
    while True:
        again = False

        rocEnv = initRocBoard()
        U = random.randint(0,U_MAX)
        if verbose:
            print "U is %d" %U

        # load the SL policies
        nnBlack = NNGoPlayer(NNGoPlayer.BLACK, sl_model, rocEnv=rocEnv)
        nnWhite = NNGoPlayer(NNGoPlayer.WHITE, sl_model, rocEnv=rocEnv)

        for i in range(U):
            if playbyplay:
                nnBlack.printRocBoard()

            # whose turn?
            nnPlayer = nnBlack if blackPlays else nnWhite
            blackPlays = not blackPlays

            # oops a game ended before reaching timestep U
            if nnPlayer.makemoveRL(playbyplay=playbyplay) or rocEnv.is_end_of_game:
                again = True
                break

        if not again:
            break

    # make a random move
    nnPlayer = nnBlack if blackPlays else nnWhite
    nnPlayer.makemoveRL(playRandom=True, playbyplay=playbyplay)
    blackPlays = not blackPlays

    if verbose:
        print "Black player is at U+1" if blackPlays else "White player is at U+1"
    uPState = nnPlayer.getState()
    uPPlayer = NNGoPlayer.BLACK if blackPlays else NNGoPlayer.WHITE

    # load the RL policies
    nnBlack = NNGoPlayer(NNGoPlayer.BLACK, rl_model, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(NNGoPlayer.WHITE, rl_model, rocEnv=rocEnv)
    # play out the game with RL policies
    while True:
        if playbyplay:
            nnBlack.printRocBoard()

        # whose turn?
        nnPlayer = nnBlack if blackPlays else nnWhite

        if nnPlayer.makemoveRL(playbyplay=playbyplay):
            winner = NNGoPlayer.WHITE if blackPlays else NNGoPlayer.BLACK
            break

        blackPlays = not blackPlays

        # did both players pass?
        if rocEnv.is_end_of_game:
            won = rocEnv.get_winner(verbose=verbose)
            if won == 0: # tie
                return
            winner = NNGoPlayer.WHITE if won==Rocgo.BLACK else NNGoPlayer.BLACK
            break

    reward = 1 if uPPlayer==winner else -1

    if verbose:
        nnBlack.printRocBoard()
        print ""
        print "U value: %d" %U
        print "Black won" if winner==NNGoPlayer.BLACK else "White won"
        print "u+1 Player = Black" if uPPlayer==NNGoPlayer.BLACK else "u+1 Player = White"
        print "u+1 State:"
        print np.transpose(uPState[0,:,:])
        print np.transpose(uPState[1,:,:])
        print "Reward: %d" %reward

    return uPState,reward


gymEnv = gym.make('Go9x9-v0')
gymEnv.reset()
gymEnv.render()
