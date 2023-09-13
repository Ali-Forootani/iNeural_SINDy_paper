#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:22:07 2022

@author: forootani
"""
import matplotlib.pyplot as plt

#Run these two commands to see plt.rcParams.update: i)import matplotlib ii) matplotlib.matplotlib_fname()

"To config the plots we modeify plt.rcParams.update"
"We must import this file inside the main file! # import plot_config_file.py"






plt.rcParams.update({
    "font.family":  "serif.Times New Roman",
    "font.style":   "normal",
    "font.variant": "normal",
    "font.weight":  "bold",
    "font.stretch": "normal",
    "font.size": 20,
    "legend.fancybox":  True,
    "text.usetex": True,
    #"text.latex.preamble": [r'\usepackage{amsmath,amssymb,bm,fontenc,physics,lmodern,nicefrac}']
    "pgf.rcfonts": True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [r'\usepackage{amsmath},\usepackage{amssymb},\usepackage{bm},\usepackge{fontenc},\usepackage{physics},\usepackge{lmodern},\usepackage{nicefrac}'],
    "figure.subplot.hspace": 0.25,
    })






#plt.rcParams.update({
#    "font.family":  "serif.Times New Roman",
#    "font.style":   "normal",
#    "font.variant": "normal",
#    "font.weight":  "bold",
#    "font.stretch": "normal",
#    "font.size": 16,
#    "axes.facecolor": "white",
#    "axes.edgecolor":  "black",
#    "axes.titlelocation": "center",
#    "axes.titlesize":   "xx-large",
#    "axes.titleweight":  "bold",
#    "axes.labelsize":  40,
#    "axes.labelpad":   8.0 ,
#    "axes.labelweight": "bold",
#    "axes.linewidth":  2,
#    "axes.labelcolor":  "k",
#    "axes.titlepad":  8,
#    "xtick.major.size": 3,
#    "xtick.minor.size": 1,
#    "xtick.major.width": 0.8,
#    "xtick.minor.width":  0.6,
#    "xtick.color":    "black",
#    "xtick.labelsize": 30,
#    "xtick.direction": "out",   # direction: {in, out, inout}
#    "xtick.major.pad": 2,
#    "xtick.minor.pad": 1,
#    "ytick.major.size": 4,
#    "ytick.minor.size": 2,
#    "ytick.major.width": 0.8,
#    "ytick.minor.width":  0.6,
#    "ytick.color": "black",
#    "ytick.labelsize": 25,
#    "ytick.major.pad": 6,
#    "ytick.minor.pad": 4,
#    "ytick.direction": "out",
#    "grid.linestyle": "--",
#    "grid.linewidth": 0.8,
#    "grid.alpha":  1.0,
#    "legend.facecolor": "inherit",
#    "figure.titlesize": "xx-large",  #xx-small, x-small, small, medium, large, x-large, xx-large, smaller, larger.
#    "figure.figsize": (12, 5),
#    "figure.dpi": 100,
#    "figure.facecolor": "w",
#    "figure.edgecolor":  "k",
#    "figure.frameon":   True,
#    "legend.fancybox":  True,
#    "legend.fontsize":  "xx-large", ##xx-small, x-small, small, medium, large, x-large, xx-large, smaller, larger.
#    "lines.linewidth": 4,
#    "lines.linestyle": "-",
#    "lines.color":  "k",
#    "lines.marker": None,
#    "lines.markerfacecolor": "auto",        # the default marker face color
#    "lines.markeredgecolor": "auto",        # the default marker edge color
#    "lines.markeredgewidth": 3,         # the line width around the marker symbol
#    "lines.markersize":    6,           # marker size, in points
#    "lines.dash_joinstyle":  "round",       # {miter, round, bevel}
#    "lines.dash_capstyle":   "butt",        # {butt, round, projecting}
#    "lines.solid_joinstyle": "round",       # {miter, round, bevel}
#    "lines.solid_capstyle":  "projecting",  # {butt, round, projecting}
#    "text.usetex": True,
#    #"text.latex.preamble": [r'\usepackage{amsmath,amssymb,bm,fontenc,physics,lmodern,nicefrac}']
#    "pgf.rcfonts": True,
#    "pgf.texsystem": "pdflatex",
#    "pgf.preamble": [r'\usepackage{amsmath},\usepackage{amssymb},\usepackage{bm},\usepackge{fontenc},\usepackage{physics},\usepackge{lmodern},\usepackage{nicefrac}'],
#    "figure.subplot.hspace": 0.25,
#    })
