# cs238_final_project DeepGo.py
An attempt at replicating [AlphaGo](https://deepmind.com/research/alphago/) by DeepMind.

## Files

### Roc_.py files
Taken from the [Rochester-NRT/RocAlphaGo project](https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/go.py).

### DataGen.py
Implements the data generation functionalities for RL and Value iteration stages.
Functions `Gym_DataGen(policyModel)`, `RL_DataGen(policyModel, opponentModel)`, and `valueDataGen(sl_model, rl_model, U_MAX)` implements 1 pass through of a simulation, and returns appropriate data for that simulation.
Functions `RL_Playout(numGames, filename, policyModel, opponentModel)` and `Value_Playout(numGames, filename, sl_model, rl_model, U_MAX)` wraps around those functions `numGames` times and stores the result to an `.hdf5` file specified via `filename`.

.hdf5 file contents for each functions are as follows:
`RL_Playout()` - 'states' 'actions' 'rewards' (actions not 1-hot encoded)
`Value_Playout()` - 'states' 'rewards'

### NNGoPlayer.py
Implements the Go player class.

Important Fields:
  * self.states - A list of all states encountered while playing
  * self.actions - A list of all actions made
  * self.nnmodel - NN backend that makes the decision
  * self.color - NNGoPlayer.BLACK or NNGoPlayer.WHITE
  * self.rocColor - Rocgo.BLACK or Rocgo.WHITE
  * self.pachiColor - pachi_py.BLACK or pachi_py.WHITE

Important Functions:
  * `makemoveGym()`
  * `makemoveRL(playRandom)`
  * `makeRandomValidMove()`

`nn_vs_nnGame(rocEnv, playBlack, nnBlack, nnWhite)` is also implemented, and it plays out a game between two NNGoPlayer classes starting at the board configuration specified in `rocEnv`

### utils.py
Implements I/O related functions.
Useful Functions:
  * `write2hdf5(filename, dict2store)`
  * `hdf52dict(hdf5Filename)`

### rochesterWrappers.py
Wrapper functions for the Rochester Go Board implementations.
Useful Functions:
  * `initRocBoard()`
  * `rocBoard2State(rocEnv)`
  * `printRocBoard(rocEnv)`
  * `returnRocBoard(rocEnv)`
  * `get_legal_coords(rocEnv)`
  * `intMove2rocMove(rocEnv)`

### MCTS.py
A Monte-Carlo Tree Search implementation. Class `MCNode` represents a node in a tree. `MCTreeSearch()` can be called to initiate the search.
