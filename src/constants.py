"""
constants.py
Isaac Jung

This module simply contains all the constants used by other modules.
"""

from json import load

DISPLAY_WIDTH = 1208  # game window width, must be at least 808 to match board.png
DISPLAY_HEIGHT = 828  # game window height, must be at least 808 to match board.png
GAME_WINDOW_ICON = 'assets/pngs/icon_small.png'  # filepath of icon that displays in upper corner of game window
GAME_WINDOW_TITLE = 'The Duke'  # string that displays as title of game window

BOARD = 'assets/pngs/board.png'  # filepath of png for the game board
BOARD_SIZE = 808  # board width and height

with open('data/tiles/types.json') as f:
    TILE_TYPES = load(f)  # data structure listing all tile types
TILE_SIZE = 128  # tile width and height, must be small enough to fit within a single space on the board
STARTING_TROOPS = ['Duke', 'Footman', 'Footman']
with open('data/tiles/movements.json') as f:
    TROOP_MOVEMENTS = load(f)  # data structure listing all troop movements
