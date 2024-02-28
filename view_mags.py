#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:59:00 2023

@author: abithasrinivas
"""

import sys
sys.path.append("..")

import magsimulator
import magcadexporter
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cuboid, CylinderSegment
import itertools
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cProfile
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, freeze_support
from os import getpid
import time
import numpy.matlib
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

filename= 'test_noconstraint_SAS_12mags_500iterations.xlsx'

mag_vect = [1270,0,0] # set mag_vector? 
ledger, magnets = magsimulator.load_magnet_positions(filename, mag_vect) # load magnet positions 
col_sensors = magpy.Collection(style_label='sensors') # initialize collection of secors to map the field 
sensor1 = magsimulator.define_sensor_points_on_sphere(20,50,[0,0,0]) # define sensor points on a shell of DSV 

col_sensors.add(sensor1) # add/append sensors on shell 
magnets = magpy.Collection(style_label='magnets') # initialize magnets 

magsimulator.plot_magnets3D(ledger)


eta, meanB0, col_magnet, B = magsimulator.simulate_ledger(magnets,col_sensors,
                                                          mag_vect,ledger,0.06,4,
                                                         True,False,None,False) # simulate the magnets on sensor positions , magnets and their positions 
print('mean B0='+str(round(meanB0,3)) +  ' homogeneity=' + str(round(eta,3))) # print

#magsimulator.plot_3D_field(B,Bcomponent=0,xmin=-100,xmax=100,ymin=-100,ymax=100,zmin=-100, zmax=100)
