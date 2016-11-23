import sgf
import os
import h5py
import numpy as np

# map from numerical coordinates to letters used by SGF
SGF_POS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BOARD_SZ = 9

def sgfWriter(actions, filename, boardSz=9):
    """
    Creates an SGF file from the input 'actions'

    @type   actions     :   (string, int)
    @param  patientID   :   Records the actions taken in a game
                            String should be 'B' for black or 'W' for white
                            Int should be the location of the action, zero-indexed
                            from the top left, row-major order.
    @type   filename    :   String
    @param  filename    :   Name of the file to save the sgf file to.
    @type   boardSz     :   int
    @param  boardSz     :   Size of the board. Defaults to 9.
    """
    sgfCollector = sgf.Collection()
    gtree = sgf.GameTree(sgfCollector)

    # characterize the first node
    # properties field for the first node characterizes the game!
    # 'SZ' - Go board size, 'EV' - event name, 'PW' - player white
    # 'KM' - komi, 'PB' - player black, etc...
    initNode = sgf.Node(gtree, previous=None)
    initNode.properties = {'SZ': [str(boardSz)], 
                           'PW': ['Yuki Inoue'], 
                           'PB': ['Yuki Inoue']}
    initNode.first = True

    prevNode = initNode
    for playColor,hand in actions:
        currNode = sgf.Node(gtree, previous=None)
        sgfHand = SGF_POS[hand%BOARD_SZ]+SGF_POS[hand/BOARD_SZ]
        currNode.properties = {playColor: [sgfHand]}

        prevNode.next = currNode
        currNode.previous = prevNode
        
        gtree.nodes.append(prevNode)
        prevNode = currNode

    gtree.nodes.append(currNode)

    sgfCollector.children.append(gtree)

    with open(filename, "w") as f:
        sgfCollector.output(f)

def sgf2hdf5(filename, sgfDir, boardSz=BOARD_SZ):
    """
    Converts sgf files to an hdf5 file

    @type   filename    :   String
    @param  filename    :   Name of the file to save the hdf5 file to.
    @type   sgfDir      :   String
    @param  sgfDir      :   Name of the directory where the sgf files are stored.
    @type   boardSz     :   int
    @param  boardSz     :   Size of the board. Defaults to 9.
    """
    preprocess_cmd_temp = 'python -m game_converter --outfile %s --directory %s --size %d --features all'
    preprocess_cmd = preprocess_cmd_temp %(filename,sgfDir,boardSz)
    os.system(preprocess_cmd)

def sgf2stateaction(sgfFilename, boardSz=BOARD_SZ):
    """
    Converts an SGF file of a game and returns a tuple of state and action pairs

    @type   sgfFilename :   String
    @param  sgfFilename :   Filename of the sgf file.
    @type   boardSz     :   int
    @param  boardSz     :   Size of the board. Defaults to 9.

    @rtype              :   A tuple of numpy matrices
    @return             :   np_states stores the state information at each time slot
                            np_acts stores the action information at each time slot
                            Dimensionalities:
                                np_states - (npositions)*48*boardSz*boardSz
                                np_acts - (npositions)*2, where 2 corresponds to
                                          x and y coordinates
    """
    filename = 'tmp.hdf5'
    preprocess_cmd_temp = 'find "%s" | python -m game_converter --outfile %s --size %d --features all'
    preprocess_cmd = preprocess_cmd_temp %(sgfFilename, filename, boardSz)
    os.system(preprocess_cmd)

    with h5py.File(filename,'r') as hf:
        np_states = np.array(hf.get("states"))
        np_acts = np.array(hf.get("actions"))

    os.remove(filename)
    return (np_states, np_acts)
