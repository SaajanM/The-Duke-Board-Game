from typing import Any, Callable
from src.game import Game
from src.learning_agent import ReinforcementLearningAgent

import numpy as np

class FeatureExtractor:
    """Static class that turns a state action pair into feature vector
    """
    def eval(state: Any, action: Any) -> np.ndarray:
        """Transform a state action pair to a feature vector

        :param state: The current state
        :param action: The action to be taken
        :return: the "goodness" of the given state action pair (the Q value)
        """
        raise NotImplementedError
    def shape() -> np.ndarray:
        """Return the shape of the feature vector

        :return: Shape of the feature vector
        """
        raise NotImplementedError

class ApproximateSarsaAgent(ReinforcementLearningAgent):
    """This agent operates based on the approximate episodic, semi-gradient sarsa approach described in Sutton and Barto pg. 244

    This class makes some modifications. Namely: Linear function approximation and epsilon greedy action decisions.

    Parameters
    ----------
    side : int
        Integer ID of the player, corresponding to which side they're on.
        There is no such thing as a player 0, so this should start at 1 and increment from there.
        Must be unique from the other player(s) in the game.
    game : Game object
        Reference to own Game in which the AI was instantiated.
        Needed so that the AI can ask the game to carry out some computations for it, such as pretend moves.
    learning_rate: float
        The speed at which the agent learns. Faster is not always better.
        A value between 0 (no learning) and 1 (1+1=2 so therefore the universe is green)
    discount_rate: float
        The amount to which the agent takes into consideration future events.
        A value between 0 (no consideration) and 1 (just as important as present)
    """

    weight_vec: np.ndarray = None
    prior_state = None
    prior_action = None

    learning_rate: float = None
    discount_rate: float = None
    epsilon: float = None
    feature_extractor: FeatureExtractor = None

    def __init__(self, side: int, game: Game, learning_rate: float, discount_rate: float, epsilon: float, feature_extractor: FeatureExtractor):
        super().__init__(side, game)
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon

        self.weight_vec = np.zeros(self.feature_extractor.shape())

    def take_turn(self):
        # Get reward for last state action pair
        # Observe new state
        # If new state is terminal do one final update (may need game.py to be rewritten)
        # Else choose new action A'
        # Do update
        # Remember old state
        # Take action A
        return super().take_turn()
    
    def __choose_action(self, state):
        """Choose an action epsilon greedily.
        
        That is, make a random action with epsilon probability otherwise go with the best action
        with probability (1-epsilon)

        :param state: The current state
        :return: the epsilon greedy action
        """
        legal_moves = self.__initialize_choice_list()
        np.random.shuffle(legal_moves)

        rng = np.random.rand()

        if rng >= self.epsilon: 
            return self.__get_best_action(state)
        else:
            return np.random.choice(legal_moves)

    def __get_best_action(self, state):
        """Calculate the best action given the current weight vector

        :param state: The current state
        :return: the "optimal" action
        """
        pass

    def __calculate_q(self, state, action) -> float:
        """Get the q value (measure of goodness) for a given state action pair via the approximation
        
        :param state: The current state
        :param action: The action to be taken
        :return: the "goodness" of the given state action pair (the Q value)
        """
        return np.ndarray.dot(self.weight_vec, self.feature_extractor.eval(state,action))

    def _calclulate_gradient(self,state,action) -> np.ndarray:
        """Get the gradient of the q function with respect to the weights

        :param state: The current state
        :param action: The action to be taken
        :return: the gradient of the "goodness" of the given state action pair (the Q value)
        """  
        return self.feature_extractor.eval(state,action)
