"""
learning_agent.py
Saajan Maslanka

This module is meant to be a container for useful base classes for learning agents.
"""

from src.ai import AI
from game import Game
from src.game import Game


class LearningAgent(AI):
    """Base Agent class. Only assumed to be some type of AI.


    Parameters
    ----------
    side : int
        Integer ID of the player, corresponding to which side they're on.
        There is no such thing as a player 0, so this should start at 1 and increment from there.
        Must be unique from the other player(s) in the game.
    game : Game object
        Reference to own Game in which the AI was instantiated.
        Needed so that the AI can ask the game to carry out some computations for it, such as pretend moves.
    """

    __game: Game = None

    def __init__(self, side: int, game: Game):
        super().__init__(side, '{} CPU'.format(self.__class__.__name__))
        self.__game = game
    
    def setup_phase(self):
        return super().setup_phase()
    
    def take_turn(self):
        raise NotImplementedError
    

class ReinforcementLearningAgent(LearningAgent):
    """(Placeholder) Base class for reinforcement learning. Assumed to operate based on some reinforcement learning approach.

    This class assumes a state action reward loop, making no assumptions about how they are stored.
    Also assumed is that the environment is episodic.

    Parameters
    ----------
    side : int
        Integer ID of the player, corresponding to which side they're on.
        There is no such thing as a player 0, so this should start at 1 and increment from there.
        Must be unique from the other player(s) in the game.
    game : Game object
        Reference to own Game in which the AI was instantiated.
        Needed so that the AI can ask the game to carry out some computations for it, such as pretend moves.
    """

    def __init__(self, side: int, game: Game):
        super().__init__(side, game)

    def setup_phase(self):
        raise NotImplementedError
    
    def take_turn(self):
        raise NotImplementedError