"""
game.py
Isaac Jung

This module contains all the code related to the UI.
"""

from pygame import display, font, Surface
from src.constants import (BG_COLOR_LIGHT_MODE, TEXT_COLOR_LIGHT_MODE, BG_COLOR_DARK_MODE, TEXT_COLOR_DARK_MODE,
                           TEXT_FONT_SIZE)
from enum import Enum


class Theme(Enum):
    """Enum for light/dark mode themes
    """
    LIGHT = 0
    DARK = 1

    def __invert__(self):
        return Theme(self.value ^ 1)


class Display:
    """Wrapper for pygame.display that includes the theme and some simpler interfaces.

    Parameters
    ----------
    width : int
        Width of the window in pixels.
    height : int
        Height of the window in pixels.
    text_font : pygame.font.Font (optional; None by default)
        Font to be used for all text in the game.
        When not provided, defaults to the pygame default font.
    theme : Theme attribute (optional; Theme.LIGHT by default)
        Affects background and text colors used.
        When not provided, defaults to a light mode theme.
    """

    def __init__(self, width, height, text_font=None, theme=Theme.LIGHT):
        self.__game_display = display.set_mode((width, height))
        self.__font = font.Font(font.get_default_font(), TEXT_FONT_SIZE)
        if text_font is not None and isinstance(text_font, font.Font):
            self.__font = text_font
        self.__theme = theme

    def toggle_theme(self):
        self.__theme = ~self.__theme

    @property
    def theme(self):
        return self.__theme

    def draw_bg(self):
        self.__game_display.fill(BG_COLOR_DARK_MODE if self.__theme == Theme.DARK else BG_COLOR_LIGHT_MODE)

    def blit(self, surface, location):
        self.__game_display.blit(surface, location)

    def write(self, text, location, right_align=False):
        """Uses the blit function to write a string to the screen.

        It first draws a blank surface over the existing area, so that layers of text don't get jumbled together.

        :param text: string to be written to the screen
        :param location: (x, y)-coordinates of the pixel location of either upper-left or upper-right of text
            The right_align parameter affects whether it will be treated as the upper-left or upper-right.
        :param right_align: boolean that determines whether the location parameter represents upper-left or upper-right
            False by default, setting it to True will have it treat location as the upper-right.
        """
        text_surface = font.Font.render(self.__font, text, True,
                                        TEXT_COLOR_DARK_MODE if self.__theme == Theme.DARK else TEXT_COLOR_LIGHT_MODE)
        bg = Surface((text_surface.get_width() + 2, text_surface.get_height()))
        bg.fill(BG_COLOR_DARK_MODE if self.__theme == Theme.DARK else BG_COLOR_LIGHT_MODE)
        if right_align:
            self.blit(bg, (location[0]-bg.get_width(), location[1]))
            self.blit(text_surface, (location[0]-text_surface.get_width(), location[1]))
        else:
            self.blit(bg, location)
            self.blit(text_surface, location)

    def draw(self, surface, location):
        """Uses the blit function to draw an image to the screen.

        It first draws a blank surface over the existing area, so that layers of images don't get jumbled together.

        :param surface: pygame.surface.Surface object containing the image to be drawn to the screen
        :param location: (x, y)-coordinates of the pixel location of upper left corner of tile
        """
        bg = Surface((surface.get_width(), surface.get_height()))
        bg.fill(BG_COLOR_DARK_MODE if self.__theme == Theme.DARK else BG_COLOR_LIGHT_MODE)
        self.blit(bg, location)
        self.blit(surface, location)
