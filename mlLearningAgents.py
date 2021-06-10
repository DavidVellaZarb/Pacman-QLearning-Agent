# The agent here was written by David Vella Zarb.
#
# The code for running Pacman was developed at UC Berkeley.
#
# As required by the licensing agreement:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from pacman import Directions
from game import Agent
import random
import game
import util


class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 2000):
        """Constructor, called when we start running the game.

        Parameters
        ==========
        alpha : float
            The learning rate.

        epsilon : float
            The exploration rate.

        gamma : float
            The discount factor.

        numTraining : float
            The number of training episodes.
        """

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        
        # Counts the number of games we have played
        self.episodesSoFar = 0
        
        self.previousScore = 0
        self.previousState  = None
        self.previousAction = None
        self.previousReward = None
        # The keys of this dictionary will be states.  Each state will have its own dictionary for actions and their Q values
        self.Q = {} 

    
    # Accessor functions for the variable episodesSoFar, controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getNumTraining(self):
        return self.numTraining

    def getMaxAttempts(self):
        return self.maxAttempts


    def maxQValue(self, state):
        """Returns the highest Q value for the actions available in the current state.  Used in the update
        equation for Q values.

        Parameters
        ==========
        state : string
            The state for which to check actions.
        """
        
        # If no Q values have been recorded for this state then return 0
        if state not in self.Q:
            return 0
        # Find the highest Q value.
        highest = -999999999
        for key in self.Q[state].keys():
            if self.Q[state][key] > highest:
                highest = self.Q[state][key]
        return highest

    def bestAction(self, state, legalMoves):
        """Returns the action with the highest Q value given a state.  If more than one action has the highest
        Q value, then it picks from those actions at random.

        Parameters
        ==========
        state : string
            The state for which to check actions.

        legalMoves : array
            The possible moves from the current position.
        """
        
        # If no Q values have been recorded for this state then pick a random action
        if state not in self.Q:
            return random.choice(legalMoves)
        # Find best action
        bestQValue = self.maxQValue(state)
        actions = [] # Used in case more than one action has the highest Q value
        for key in self.Q[state].keys():
            if self.Q[state][key] == bestQValue:
                actions.append(key)
        bestAction = random.choice(actions)
        return random.choice(actions)

    def initialiseStateAction(self, state, action):
        """Checks if the given state-action pair exists in Q, and if not initialises it as 0.

        Parameters
        ==========
        state : string
            The state to be checked in Q.

        action : string
            The action to be checked in Q.
        """
        
        Q = self.Q
        if state not in Q:
            Q[state] = {}
        if action not in Q[state]:
            Q[state][action] = 0

    def QLearningIteration(self, state, reward, legalMoves, terminal=False):
        """Implements the Q-learning algorithm in the lecture slides.  Updates the Q value for the previous state (and the current state
        if it is a terminal state).  Returns an action using the epsilon-greedy method.

        Parameters
        ==========
        state : string
            A string representing the current state.  Contains Pacman, ghost and food positions.

        reward : int
            The reward received from the current state.

        legalMoves : array
            The possible moves from the current position.

        terminal : boolean, optional
            A flag indicating whether the current state is a terminal state.  Default is false.
        """
        
        Q = self.Q
        # Update Q value
        if terminal:
            if state not in Q:
                Q[state] = {}
            Q[state]['None'] = reward
        if (self.previousState != None):
            self.initialiseStateAction(self.previousState, self.previousAction) # If state-action pair is not in Q, then initialise it as 0.
            Q[self.previousState][self.previousAction] = float(Q[self.previousState][self.previousAction]) + (self.alpha * (float(self.previousReward) + (self.gamma * float(self.maxQValue(state))) - float(Q[self.previousState][self.previousAction])))
        # Update previous state and reward to the current ones
        self.previousState = state
        self.previousReward = reward
        # Epsilon-greedy action selection
        randomNumber = random.random()
        threshold = 1 - self.epsilon
        # Greedy action
        if randomNumber < threshold:
            self.previousAction = self.bestAction(state, legalMoves)
            return self.previousAction
        if len(legalMoves) == 0: # Used when current state is a terminal state
            return 'None'
        # Random action
        self.previousAction = random.choice(legalMoves)
        return self.previousAction

    def encodeState(self, position, ghostLocations, foodLocations):
        """Represents state information (Pacman's position, ghost and food locations) as a string and returns it.

        Parameters
        ==========
        position : tuple
            Pacman's current position.

        ghostLocations : array
            Current ghost locations, each stored as a tuple.

        foodLocations : array
            Current food locations, stored as a 2D array of booleans.
        """
        
        output = str(position[0]) + str(position[1])
        for ghost in ghostLocations:
            output = output + str(int(ghost[0])) + str(int(ghost[1]))
        for row in foodLocations:
            for column in row:
                if str(column) == 'False':
                    output = output + 'F'
                else:
                    output = output + 'T'
        return output


    def getAction(self, state):
        """The main method required by the game.  Called every time that Pacman is expected to move.

        Parameters
        ==========
        state : array
            A grid storing the current locations of all objects (ghosts, food, walls and capsules) in
            the game.
        """
   
        # Remove 'STOP' from legal actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Represent the current state as a string
        stateEncoding = self.encodeState(state.getPacmanPosition(), state.getGhostPositions(), state.getFood())
            
        # Set reward as the difference between the current score and the last
        reward = state.getScore() - self.previousScore
        self.previousScore = state.getScore()
            
        # Run one iteration of the Q-learning algorithm and perform the action returned
        return self.QLearningIteration(stateEncoding, reward, legal)         

    def final(self, state):
        """Handles the end of episodes.  This is called by the game after a win or a loss.

        Parameters
        ==========
        state : array
            A grid storing the current locations of all objects (ghosts, food, walls and capsules) in
            the game.
        """

        # Represent the current state as a string
        stateEncoding = self.encodeState(state.getPacmanPosition(), state.getGhostPositions(), state.getFood())

        # Set reward as the difference between the current score and the last
        reward = state.getScore() - self.previousScore

        # Run one iteration of the Q-learning algorithm to update the Q value for the terminal state
        self.QLearningIteration(stateEncoding, reward, state.getLegalPacmanActions(), True)
        
        # Keep track of the number of games played, and set learning parameters to zero when we are done
        # with the pre-set number of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            print(len(self.Q))
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


