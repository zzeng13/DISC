import matplotlib
import matplotlib.cm
import numpy as np


def simple_scoring_viz(words, color_array, cmap):
    # Reference: https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768#file-colorize_text-py
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words

    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string