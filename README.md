# Pacman-QLearning-Agent

This creates and trains a Q-Learning agent capable of winning all Pacman games it plays.  Note that my code is in
mlLearningAgents.py.  The rest is code to run the Pacman game, developed at UC Berkeley.

## Running Instructions

**Note:** The code for the agent uses Python 2, as the Pacman game was developed using that version.  It will not run
in Python 3 due to differences in syntax.

To run this programme, simply cd into the directory and run `py -2.7 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid`.

This trains the agent using 2000 iterations, and then plays the game 10 additional times to calculate the winning rate.
This game is played in a small grid (to have a reasonable running time) - if you wish to see it play in a large grid
simply remove `-l smallGrid` from the above command and give it a larger number of training iterations.
