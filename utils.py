import sgf
import os
import h5py
import numpy as np
from Rocgame_converter import *

# constants
# map from numerical coordinates to letters used by SGF
SGF_POS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BOARD_SZ = 9
# default feature list for the state generation
FEATURE_LIST = ["board", "ones", "turns_since", "liberties", "capture_size",
                "self_atari_size", "liberties_after", "ladder_capture", 
                "ladder_escape", "sensibleness", "zeros"]

def sgfWriter(actions, filename, boardSz=9):
    """
    Creates an SGF file from the input 'actions'

    @type   actions     :   Integer List
    @param  actions     :   Records the actions taken in a game
                            Each integer should be the location of the action, zero-indexed
                            from the top left, row-major order.
                            Black goes first, as specified via go standard rule.
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
    playColors = ['B','W']
    count = 0
    for act in actions:
        currNode = sgf.Node(gtree, previous=None)
        sgfAct = SGF_POS[act%BOARD_SZ]+SGF_POS[act/BOARD_SZ]
        currNode.properties = {playColors[count]: [sgfAct]}

        prevNode.next = currNode
        currNode.previous = prevNode
        
        gtree.nodes.append(prevNode)
        prevNode = currNode

        count = (count+1)%2

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
    run_game_converter(['--outfile', filename, '--directory', sgfDir, '--size', str(boardSz), '--features', 'all'])

def hdf52stateaction(hdf5Filename):
    """
    Loads an HDF5 file of a game and returns a tuple of state and action pairs

    @type   hdf5Filename:   String
    @param  hdf5Filename:   Filename of the hdf5 file.

    @rtype              :   A tuple of numpy matrices
    @return             :   np_states stores the state information at each time slot
                            np_acts stores the action information at each time slot
                            Dimensionalities:
                                np_states - (npositions)*48*boardSz*boardSz
                                np_acts - (npositions)*2, where 2 corresponds to
                                          x and y coordinates
    """
    with h5py.File(hdf5Filename,'r') as hf:
        np_states = np.array(hf.get("states"))
        np_acts = np.array(hf.get("actions"))

    return (np_states, np_acts)

def sgf2stateaction(filename, boardIndx, feature_list=FEATURE_LIST):
    """
    Loads an SGF file and returns a tuple of state and action pairs

    @type   filename    :   String
    @param  filename    :   Name of the SGF file.
    @type   boardIndx   :   Integer List
    @param  boardIndx   :   List of board locations to extract the state-action pairs from.
                            0 indexed, time slice 0 refers to the blank board. 
                            In other words, the first action (Black) occurs at index 0.
    @type   feature_list:   String List
    @param  feature_list:   State features. Refer to FEATURE_LIST for how to specify them.

    @rtype              :   A tuple of numpy matrices
    @return             :   states stores the state information at each time slot
                            actions stores the action information at each time slot
                            Empty tuple returned when error
                            Dimensionalities:
                                states - (npositions)*48*boardSz*boardSz
                                actions - (npositions)*2, where 2 corresponds to
                                          x and y coordinates
    """
    converter = game_converter(feature_list)

    states = np.empty([len(boardIndx),48,BOARD_SZ,BOARD_SZ])
    actions = np.empty([len(boardIndx),1])

    indx = 0
    count = -1
    try:
        for state,action in converter.convert_game(filename, bd_size=BOARD_SZ):
            count += 1
            if count not in boardIndx:
                continue
            states[indx,:,:,:] = state
            actions[indx,:] = action[0]+action[1]*BOARD_SZ
            indx += 1

        if count<max(boardIndx):
            warnings.warn("%d is larger than how many hands were played in %s; " % (max(boardIndx),filename))
            return ()

    except go.IllegalMove:
        warnings.warn("Illegal Move encountered in %s\n"
                      "\tdropping the remainder of the game" % filename)
        return ()
    except sgf.ParseException:
        warnings.warn("Could not parse %s\n\tdropping game" % filename)
        return ()
    except SizeMismatchError:
        warnings.warn("Skipping %s; wrong board size" % filename)
        return ()
    except Exception as e:
        # catch everything else
        warnings.warn("Unkown exception with file %s\n\t%s" % (filename, e),
                      stacklevel=2)
        return ()

    return (states,actions)

# Usage:
# sgfWriter([0,1,2,3,4,9,10],'example.sgf')
# s,a = sgf2stateaction('example.sgf',[0,1,2,3,4,5,6])
# print s[6,0:3,:,:]
# print a