import numpy as np
import copy
import math

import Rocgo
from constants import *
from NNGoPlayer import NNGoPlayer

from rochesterWrappers import *

class MCNode(object):
    """
    Implements the 
    """

    nVl = 3     # "virtual loss"
    nThr = 40   # expansion threshold
    lmbda = 0.5 # value score/rollout score mixing constant
    beta = 0.67 # prior softmax constant
    cpuct = 5   # exploration term

    def __init__(self, rocEnv, P, color, sl_nnModel):
        # AKA Expansion
        self.rocEnv = rocEnv
        self.P = P
        self.color = color

        self.actionSpace = BOARD_SZ**2+2 # plus 2 for pass and resign actions
        self.Nv = [0]*self.actionSpace
        self.Nr = [0]*self.actionSpace
        self.Wv = [0]*self.actionSpace
        self.Wr = [0]*self.actionSpace
        self.Q = [0]*self.actionSpace

        self.sl_nnModel = sl_nnModel
        self.prior = self.getPrior()

        # TODO: check if deepcopy works with Rochester Board Object
        nextColor = Rocgo.WHITE if self.color==Rocgo.BLACK else Rocgo.BLACK
        self.nextStates = [None]*self.actionSpace
        validMoves = get_legal_coords(self.rocEnv)
        for act in range(self.actionSpace-1):
            if act in validMoves:
                self.nextStates[act] = copy.deepcopy(self.rocEnv)
                self.nextStates[act].do_move(intMove2rocMove(act), color=nextColor)
            else:
                self.Q[act] = -float('inf')

        self.children = [None]*self.actionSpace

    def getPrior(self):
        # TODO
        # use rocBoard2State() and self.sl_nnModel and beta to figure out the prior
        return [0.5]*self.actionSpace

    def update(self, act, gameResult, vAtLeaf):
        self.Nr[act] += 1
        self.Nv[act] += 1
        self.Wr[act] += gameResult
        self.Wv[act] += vAtLeaf

        self.Q[act] = (1-MCNode.lmbda)*self.Wv[act]/self.Nv[act] + MCNode.lmbda*self.Wr[act]/self.Nr[act] 

        # expand
        if self.Nr[act]==MCNode.nThr:
            self.expand(act)

    def expand(self, act):
        nextColor = Rocgo.WHITE if self.color==Rocgo.BLACK else Rocgo.BLACK
        self.children[act] = MCNode(self.nextStates[act], self.prior[act], nextColor, self.sl_nnModel)

    def __str__(self):
        repStr = ''
        repStr += 'State:\n'
        repStr += str(returnRocBoard(self.rocEnv))+'\n'
        repStr += 'P:\n'
        repStr += str(self.P)+'\n'
        repStr += 'Color:\n'
        repStr += "Black\n" if self.color==Rocgo.BLACK else "White\n"
        repStr += 'Nv:\n'
        repStr += str(self.Nv)+'\n'
        repStr += 'Nr:\n'
        repStr += str(self.Nr)+'\n'
        repStr += 'Wv:\n'
        repStr += str(self.Wv)+'\n'
        repStr += 'Wr:\n'
        repStr += str(self.Wr)+'\n'
        repStr += 'Q:\n'
        repStr += str(self.Q)+'\n'
        repStr += 'prior:\n'
        repStr += str(self.prior)+'\n'
        return repStr

def reachLeaf(root):
    nodes = []
    acts = []
    nodes.append(root)
    currNode = root
    # reach a leaf
    while True:
        # choose an action
        act = None
        u_num = MCNode.cpuct*currNode.P*math.sqrt(sum(currNode.Nr))
        for i in range(currNode.actionSpace):
            u = u_num/(1+currNode.Nr[i])
            val = currNode.Q[i]+u
            if not act or maxVal<val:
                act = i
                maxVal = val

        acts.append(act)

        nextNode = currNode.children[act]
        if not nextNode:
            break

        nodes.append(nextNode)
        currNode = nextNode

    return nodes,acts


def getVAtLeaf(value_nnModel, state):
    # TODO
    # get value at a state using the NN Model
    return 0.5

def rollout_makemove(state):
    return 1

def playout(rolloutModel, rocEnv, verbose=False):
    # playout logic, returns +1 for winning, -1 for losing
    nnBlack = NNGoPlayer(NNGoPlayer.BLACK, rolloutModel, rocEnv=rocEnv)
    nnWhite = NNGoPlayer(NNGoPlayer.WHITE, rolloutModel, rocEnv=rocEnv)

    leafColor = NNGoPlayer.BLACK if rocEnv.current_player==Rocgo.BLACK else NNGoPlayer.WHITE
    winner = nn_vs_nnGame(rocEnv, rocEnv.current_player==Rocgo.BLACK, nnBlack, nnWhite)

    # end of the game tasks
    # determine who the winner is
    reward = 1 if (leafColor == winner) else -1

    if verbose:
        printRocBoard(nnBlack.rocEnv)
        print ""
        print "Winner: %s" %("Black Player" if winner==NNGoPlayer.BLACK else "White Player")
        print "Leaf color: %s" %("Black" if leafColor==NNGoPlayer.BLACK else "White")
        print "Did I win? %d" % reward

    # return the game result
    return reward

def update(nodes, acts, result, vAtLeaf):
    for node,act in zip(nodes, acts):
        node.update(act, result, vAtLeaf)

MCTS_SEARCH_NUM = 1
def MCTreeSearch(state, color, sl_nnModel, rolloutModel, value_nnModel):
    root = MCNode(state, 1, Rocgo.WHITE if color==Rocgo.BLACK else Rocgo.BLACK, sl_nnModel)

    # expand out the nodes for the first move
    for i in range(BOARD_SZ**2+1):
        print i
        root.expand(i)
    root.Nr = [MCNode.nThr]*root.actionSpace

    # explore the actions & refine Q
    for i in range(MCTS_SEARCH_NUM):
        # play the game up till the leaf
        nodes,acts = reachLeaf(root)

        # find the value at leaf node
        boardAtLeaf = nodes[-1].nextStates[acts[-1]]
        vAtLeaf = getVAtLeaf(value_nnModel, rocBoard2State(boardAtLeaf))

        # play the rest of the game
        result = playout(rolloutModel, boardAtLeaf)

        update(nodes, acts, result, vAtLeaf)

    print root

    # choose the best Q
    return np.argmax(np.asarray(root.Q))

rocEnv = initRocBoard()
MCTreeSearch(rocEnv, Rocgo.BLACK, None, None, None)