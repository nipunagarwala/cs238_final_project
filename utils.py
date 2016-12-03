import sgf
import h5py
import numpy as np
from Rocgame_converter import *
import Rocgo as go
from constants import *

# constants
# map from numerical coordinates to letters used by SGF
SGF_POS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def sgfWriter(actions, filename, boardSz=BOARD_SZ):
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

        if act==PASS_ACTION:
            currNode.properties = {playColors[count]: ['']}
        elif act!=RESIGN_ACTION:
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
    run_game_converter(['--outfile', filename, '--directory', sgfDir, '--size', str(boardSz),
                        '--features', 'all', '--recurse'])

def write2hdf5(filename, dict2store):
    """
    Write items in a dictionary to an hdf5file

    @type   filename    :   String
    @param  filename    :   Filename of the hdf5 file to output to.
    @type   dict2store  :   Dict
    @param  dict2store  :   Dictionary of items to store. The value should be an array.

    """
    with h5py.File(filename,'w') as hf:
        for key,value in dict2store.iteritems():
            hf.create_dataset(key, data=value,compression="lzf")

def hdf52dict(hdf5Filename):
    """
    Loads an HDF5 file of a game and returns a dictionary of the contents

    @type   hdf5Filename:   String
    @param  hdf5Filename:   Filename of the hdf5 file.
    """
    retDict = {}
    with h5py.File(hdf5Filename,'r') as hf:
        for key in hf.keys():
            retDict[key] = np.array(hf.get(key))

    return retDict

def perLayerOp(mat, opFunc):
    """
    Assumes that mat is a 3D numpy matrix, and applies the function specified 
    via opFunc for dimensions 2 and 3 of mat.
    """
    retMat = np.empty_like(mat)
    for i in range(mat.shape[0]):
        retMat[i,:,:] = opFunc(mat[i,:,:])

    return retMat

def hdf5Augment(filename, outfilename):
    """
    Augments the games stored in filename by rotating and reflecting the states.

    @type   filename    :   String
    @param  filename    :   Name of the file to read from.
    """
    states = []
    actions = []

    originalDict = hdf52dict(filename)
    oriSts = originalDict['states']
    oriActs = originalDict['actions']

    count = 0
    numSamples = oriSts.shape[0]
    for i in range(numSamples):
        count += 1
        if count%1000==0:
            print count

        state0 = oriSts[i,:,:,:]
        state90 = perLayerOp(state0, np.rot90)
        state180 = perLayerOp(state90, np.rot90)
        state270 = perLayerOp(state180, np.rot90)
        state0Ref = perLayerOp(state0, np.fliplr)
        state90Ref = perLayerOp(state90, np.fliplr)
        state180Ref = perLayerOp(state180, np.fliplr)
        state270Ref = perLayerOp(state270, np.fliplr)

        states.append(state0)
        states.append(state90)
        states.append(state180)
        states.append(state270)
        states.append(state0Ref)
        states.append(state90Ref)
        states.append(state180Ref)
        states.append(state270Ref)

        actionBoard = np.zeros((BOARD_SZ,BOARD_SZ))
        actionBoard[oriActs[i][0],oriActs[i][1]] = 1
        actionBoard90 = np.rot90(actionBoard).copy()
        actionBoard180 = np.rot90(actionBoard90).copy()
        actionBoard270 = np.rot90(actionBoard180).copy()
        actionBoard0Ref = np.fliplr(actionBoard).copy()
        actionBoard90Ref = np.fliplr(actionBoard90).copy()
        actionBoard180Ref = np.fliplr(actionBoard180).copy()
        actionBoard270Ref = np.fliplr(actionBoard270).copy()

        oneLoc = np.where(actionBoard)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard90)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard180)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard270)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard0Ref)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard90Ref)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard180Ref)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)
        oneLoc = np.where(actionBoard270Ref)
        actions.append(oneLoc[0][0]+oneLoc[1][0]*BOARD_SZ)

    write2hdf5(outfilename, {'states':states, 'actions':actions})

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

import gym
import pachi_py
from rochesterWrappers import printRocBoard,returnRocBoard
from NNGoPlayer import NNGoPlayer
import sys
from cStringIO import StringIO
def pachiGameRecorder(filename, verbose=False, playbyplay=False):
    """
    Plays a game between two pachi players.
    Stores the game as an sgf file with name 'filename'
    """
    if os.path.isfile(filename):
       return
    actions = []
    ff = ''

    # create 2 gym environments
    gymEnv1 = gym.make('Go9x9-v0')
    gymEnv1.reset()
    gymEnv1.state.color = pachi_py.WHITE
    gymEnv1.player_color = pachi_py.WHITE

    gymEnv2 = gym.make('Go9x9-v0')
    gymEnv2.reset()

    # create 2 dummies to play against pachi
    dummy1 = NNGoPlayer(NNGoPlayer.WHITE, None, gymEnv=gymEnv1)
    dummy2 = NNGoPlayer(NNGoPlayer.BLACK, None, gymEnv=gymEnv2)

    # play out the game
    playBlack = True
    dummy1.last_pachi_mv = PASS_ACTION
    dummy2.last_pachi_mv = PASS_ACTION
    while True:
        if playbyplay:
            printRocBoard(dummy1.rocEnv)
            printRocBoard(dummy2.rocEnv)
        ff += 'Black playing' if playBlack else 'White playing'
        ff += '\n'
        ff += str(returnRocBoard(dummy1.rocEnv)).replace('0',' ').replace('.','')
        ff += '\n'
        ff += gymEnv1.state.board.__str__()
        ff += '\n'
        ff += 'last move: %d' %dummy1.last_pachi_mv
        ff += '\n'
        ff += str(returnRocBoard(dummy2.rocEnv)).replace('0',' ').replace('.','')
        ff += '\n'
        ff += gymEnv2.state.board.__str__()
        ff += '\n'
        ff += 'last move: %d' %dummy2.last_pachi_mv
        ff += '\n'
        ff += '\n'
        ff += '\n'

        dummyPlaying = dummy1 if playBlack else dummy2
        dummyNotPlaying = dummy2 if playBlack else dummy1
        playBlack = not playBlack

        if dummyNotPlaying.last_pachi_mv==PASS_ACTION:
            print 'passed'
        try:
            if dummyPlaying.makemoveGym(move=dummyNotPlaying.last_pachi_mv, 
                                        playbyplay=playbyplay):
                break
        except:
            print ff
            gymEnv1.render()
            printRocBoard(dummy1.rocEnv)
            gymEnv2.render()
            printRocBoard(dummy2.rocEnv)
            exit(1)

        actions.append(dummyPlaying.last_pachi_mv)

    if verbose:
        # setup the environment
        backup = sys.stdout
        # ####
        sys.stdout = StringIO()     # capture output
        gymEnv1.step(PASS_ACTION)
        gymEnv1.render()
        out = sys.stdout.getvalue() # release output
        gym1board = '\n'.join(out.split('\n')[2:])
        # ####
        sys.stdout.close()  # close the stream 
        sys.stdout = backup # restore original stdout


        # setup the environment
        backup = sys.stdout
        # ####
        sys.stdout = StringIO()     # capture output
        gymEnv2.step(PASS_ACTION)
        gymEnv2.render()
        out = sys.stdout.getvalue() # release output
        gym2board = '\n'.join(out.split('\n')[2:])
        # ####
        sys.stdout.close()  # close the stream 
        sys.stdout = backup # restore original stdout

        if gym1board!=gym2board:
            print ff
            print gym1board
            print gym2board
            printRocBoard(dummy1.rocEnv)
            printRocBoard(dummy2.rocEnv)

            print ""
            print "Winner: %s" %('Black' if gymEnv1.state.board.official_score<0 else 'White')
            print "Score: %d" % gymEnv1.state.board.official_score

    print actions
    sgfWriter(actions, filename)

def pachi_game_Dump(num_games=1000):
    """
    Runs pachiGameRecorder() 'num_games' times and dumps the resulting sgf files
    """
    from multiprocessing import Pool

    filename = 'pachi_games3/pachi_game_%d.sgf'
    p = Pool(4)
    filenames = [filename%i for i in list(range(num_games))]
    p.map(pachiGameRecorder,filenames)
    # for i in range(num_games):
    #     print i
    #     pachiGameRecorder(filename=filename%i, verbose=True,playbyplay=False)

#pachi_game_Dump(30000)
#sgf2hdf5('pachi5000.hdf5', '../godata/pachi5000', boardSz=BOARD_SZ)
#sgf2hdf5('featuresNew.hdf5', 'pachi_games', boardSz=BOARD_SZ)
#hdf5Augment('pachi5000.hdf5', 'pachi5000Augment.hdf5')