import gym
import pachi_py
import numpy as np
import random

import Rocgo
from constants import *
from NNGoPlayer import NNGoPlayer,nn_vs_nnGame
from rochesterWrappers import *
from utils import write2hdf5

def Gym_DataGen(policyModel, verbose=False, playbyplay=False):
    """
    Plays out a Gym game against Pachi.

    Returns nnPlayer (holds states and actions) and reward if a game played had a winner. 
    Returns None if the game is a tie.
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
        observation,_,_,_ = gymEnv.step(PASS_ACTION)
        nnPlayer.updatePachiMove(observation, playbyplay=playbyplay)

    # play out the game
    while True:
        if playbyplay:
            gymEnv.render()
            printRocBoard(nnPlayer.rocEnv)

        if nnPlayer.makemoveGym(playbyplay=playbyplay):
            break

    if gymEnv.state.board.official_score==0:
        if verbose:
            print 'A tie game...'
        return None

    winner = NNGoPlayer.BLACK if gymEnv.state.board.official_score<0 else NNGoPlayer.WHITE
    reward = 1 if (policyColor == winner) else -1

    if verbose:
        nnPlayer.gymEnv.render()
        printRocBoard(nnPlayer.rocEnv)
        print ""
        print "Score: %d" % gymEnv.state.board.official_score
        print "Did I win? %d" % reward
        print "Actions taken: %s" %str(nnPlayer.actions)

    return (nnPlayer,reward)

def RL_DataGen(policyModel, opponentModel, verbose=False, playbyplay=False):
    """
    Plays out an RL game.

    Returns nnPlayer (holds states and actions) and reward if a game played had a winner. 
    Returns None if the game is a tie.
    """
    rocEnv = initRocBoard()

    policyColor = random.choice([NNGoPlayer.BLACK, NNGoPlayer.WHITE])
    if verbose:
        print "Playing Black" if policyColor==NNGoPlayer.BLACK else "Playing White"

    rocEnv.current_player = Rocgo.BLACK if policyColor==NNGoPlayer.BLACK else Rocgo.WHITE

    # set the nnmodel consistent with policyColor
    blackModel = policyModel if policyColor==Rocgo.BLACK else opponentModel
    whiteModel = policyModel if policyColor==Rocgo.WHITE else opponentModel
    nnBlack = NNGoPlayer(NNGoPlayer.BLACK, blackModel, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(NNGoPlayer.WHITE, whiteModel, rocEnv=rocEnv)

    winner = nn_vs_nnGame(rocEnv, True, nnBlack, nnWhite, verbose=verbose, playbyplay=playbyplay)
    if winner==None:
        return None

    # end of the game tasks
    # determine who the winner is
    reward = 1 if (policyColor == winner) else -1

    # load the state-action pairs to learn from
    nnPlayer = nnBlack if policyColor==NNGoPlayer.BLACK else nnWhite

    if verbose:
        printRocBoard(nnPlayer.rocEnv)
        print ""
        print "Winner: %s" %("Black Player" if winner==NNGoPlayer.BLACK else "White Player")
        print "Did I win? %d" % reward
        print "Actions taken: %s" %str(nnPlayer.actions)

    return (nnPlayer,reward)

def RL_Playout(numGames, policyModel, filename=None, opponentModel=None, verbose=False, playbyplay=False):
    """
    Plays out 'numGames' Games between policyModel and opponentModel.
    If the opponentModel is None, the policy will play against Pachi under OpenAI Gym.
    Saves the state action pairs to a file if 'filename' is specified.
    """
    gamesPlayed = 0
    gamesWon = 0
    win_states = []
    win_actions = []
    lose_states = []
    lose_actions = []
    while gamesPlayed<numGames:
        if opponentModel:
            output = RL_DataGen(policyModel, opponentModel, verbose=verbose, playbyplay=playbyplay)
        else:
            output = Gym_DataGen(policyModel, verbose=verbose, playbyplay=playbyplay)
        if output:
            gamesPlayed += 1
            nnPlayer,reward = output
            if reward==1:
                gamesWon += 1
                win_states += nnPlayer.states
                win_actions += nnPlayer.actions
            else:
                lose_states += nnPlayer.states
                lose_actions += nnPlayer.actions

    if filename:
        write2hdf5(filename, {'win_states':win_states, 'win_actions':win_actions, 
                              'lose_states':lose_states, 'lose_actions':lose_actions,
                              'rewards':rewards})

    return win_states, win_actions, lose_states, lose_actions, gamesWon

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
                printRocBoard(nnBlack.rocEnv)

            # whose turn?
            currPlayer = nnBlack if blackPlays else nnWhite
            blackPlays = not blackPlays

            # oops a game ended before reaching timestep U
            if currPlayer.makemoveRL(playbyplay=playbyplay) or rocEnv.is_end_of_game:
                again = True
                if verbose:
                    print 'Dang a game ended before time U'
                break

        if not again:
            break

    # make a random move
    nnPlayer = nnBlack if blackPlays else nnWhite
    nnPlayer.makemoveRL(playRandom=True, playbyplay=playbyplay)
    blackPlays = not blackPlays

    if verbose:
        print "Black player is at U+1" if blackPlays else "White player is at U+1"
    uPState = rocBoard2State(nnPlayer.rocEnv)
    uPPlayer = NNGoPlayer.BLACK if blackPlays else NNGoPlayer.WHITE

    # load the RL policies
    nnBlack = NNGoPlayer(NNGoPlayer.BLACK, rl_model, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(NNGoPlayer.WHITE, rl_model, rocEnv=rocEnv)
    # play out the game with RL policies
    winner = nn_vs_nnGame(rocEnv, blackPlays, nnBlack, nnWhite, verbose=verbose, playbyplay=playbyplay)

    # the game was a tie
    if not winner:
        if verbose:
            print 'A tie game...'
        return None

    reward = 1 if uPPlayer==winner else -1

    if verbose:
        printRocBoard(nnBlack.rocEnv)
        print ""
        print "U value: %d" %U
        print "Black won" if winner==NNGoPlayer.BLACK else "White won"
        print "u+1 Player = Black" if uPPlayer==NNGoPlayer.BLACK else "u+1 Player = White"
        print "u+1 State:"
        print np.transpose(uPState[0,:,:])
        print np.transpose(uPState[1,:,:])
        print "Reward: %d" %reward

    return uPState,reward

def Value_Playout(numGames, sl_model, rl_model, filename=None, U_MAX=90, verbose=False, playbyplay=False):
    """
    Plays out 'numGames' value iteration games.
    Saves the state action pairs to a file if 'filename' is specified.
    """
    gamesPlayed = 0
    states = []
    actions = []
    rewards = []
    while gamesPlayed<numGames:
        output = valueDataGen(sl_model, rl_model, U_MAX=U_MAX, verbose=verbose, playbyplay=playbyplay)

        if output:
            gamesPlayed += 1
            state,reward = output
            states.append(state)
            rewards.append(reward)

    if filename:
        write2hdf5(filename, {'states':states, 'rewards':rewards})

    return states,rewards