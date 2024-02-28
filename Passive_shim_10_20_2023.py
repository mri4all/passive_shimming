
'''
Passive shimming - 20cm - 10/18/2023
'''

# imports all functions

import sys
sys.path.append("..")
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


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool
import pickle
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

## define colorbar for figures 

#%%

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", 
                              pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


## import main magnet B0 map 
#filename = '../../data/optimization_after_neonate_magnet_Rmin_132p7mm_extrinsic_rot_DSV140mm_maxh60_maxlayers6_maxmag990.xlsx'
#mag_vect = [1270,0,0] # set mag_vector? 
#ledger, magnets = magsimulator.load_magnet_positions(filename, mag_vect) # load magnet positions 
#col_sensors = magpy.Collection(style_label='sensors') # initialize collection of secors to map the field 
#sensor1 = magsimulator.define_sensor_points_on_sphere(20,50,[0,0,0]) # define sensor points on a shell of DSV 

#col_sensors.add(sensor1) # add/append sensors on shell 
#magnets = magpy.Collection(style_label='magnets') # initialize magnets 

#magsimulator.plot_magnets3D(ledger)

#eta, meanB0, col_magnet, B = magsimulator.simulate_ledger(magnets,col_sensors,
 #                                                         mag_vect,ledger,0.06,4,
  #                                                        True,False,None,False) # simulate the magnets on sensor positions , magnets and their positions 
#print('mean B0='+str(round(meanB0,3)) +  ' homogeneity=' + str(round(eta,3))) # print

# dump the data 
#fileObj = open(filename[:-4]+'pkl', 'wb')
#pickle.dump(B,fileObj)
#fileObj.close() 

## start defining problem  of GA optimization 

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    # initialization of the GA 
    def __init__(self, **kwargs):
        
        filename1 = '../../field_mapping/map4_pos.mat'
        filename2 = '../../field_mapping/map4_field.mat'
        #file = open(filename[:-4]+'pkl', 'rb')
        tmp = magsimulator.data_from_matlab(filename2)
        B_tmp =  tmp['field_matrix']  # main B0 field over which we are shimming 
        B_tmp1 = B_tmp - np.mean(B_tmp,axis=0)
        eta = 42580*np.std(B_tmp1[:,0])
        print('base b0 homogeneity',eta)
        self.B_background = B_tmp[:,0:3]
        
        tmp = magsimulator.data_from_matlab(filename1)
        self.senrsor_points = tmp['pos'] 
        
        #print(len(self.B_background))
        #print(np.mean(self.B_background[:,0]))
        self.cube_side_length = 3 # mm SAS- Length of the shim magnet side 
        # self.maxzextent = 200 # mm SAS- max length of all shim rings along the bore
        #self.rmin= 68  #baseradius-self.cube_side_length
        self.mags_per_ring = 10 # SAS- max number of magnets per shim ring 
        self.rmin=66.5
        self.zpos = np.append(np.arange (-151,-69,3), np.arange (70,154,3))
        #self.zpos = np.arange (-150,151,5)
        self.nrings = len(self.zpos)
        
        variables = dict()
        
        #variables["x01"] = Choice(options=list(np.r_[68])) #mm -SAS - optimize for a 68mm or 183mm radius - either outside rf coil or inside gradient coil
        #variables["x01"] = Choice(options=list(np.r_[np.arange (-100,120,20)])) #SAS - rows of 7mm each for 20cm along bore - 14 rings on each side of center 
        #variables["x01"] = Choice(options=list(np.r_[-np.pi/8:np.pi/6:np.pi/16])) # rotation of the ring
        
        #for k in range(3, 3+self.mags_per_ring): #displacement
        #    variables[f"x{k:02}"] = Choice(options=list(np.r_[4:5:1]))           
        
        for k in range(0, self.mags_per_ring): # zrot for each mag
            # variables[f"x{k:02}"] = Real(bounds=(-np.pi/2, np.pi/2))   
            variables[f"x{k:02}"] = Choice(options=list(np.r_[0:2*np.pi:np.pi/2])) # SAS - 90 deg rotations        

        #for k in range(self.mags_per_ring*2, self.mags_per_ring*3): # yrot for each mag
        #     # variables[f"x{k:02}"] = Real(bounds=(-np.pi, np.pi))  
        #    variables[f"x{k:02}"] = Choice(options=list(np.r_[0:2*np.pi:np.pi/2]))  # SAS - no rotaton on y                 

        # for k in range(5+self.mags_per_ring*3, 5+self.mags_per_ring*4): # xrot for each mag
        #     # variables[f"x{k:02}"] = Real(bounds=(-np.pi/8, np.pi/8)) 
        #     variables[f"x{k:02}"] = Choice(options=list(np.r_[0]))  # SAS - no rotation on x

        for k in range(self.mags_per_ring*2, self.mags_per_ring*(self.nrings+2)): # binary choice of each magnet
             variables[f"x{k:02}"] = Binary()  
        
        
        super().__init__(vars=variables,n_ieq_constr=0, n_obj=1, **kwargs)
        
    # Run GA 
    def _evaluate(self, x, out, *args, **kwargs): # is this input x different than the location x??????

        # SAS assign same definitions as in init
        #thetaoffset = np.array(x[f"x01"]).reshape((1,1))
        #thetaoffset=thetaoffset[0]
        # self.rmin = np.array(x[f"x01"]).reshape((1,1))
        rmin = self.rmin
        
        #zrows = np.array(x["x01"]).reshape((1,1))
        #zrows=int(zrows[0])
        zrows = self.zpos
        #print(zrows)
        #magnet_displacement=np.array([x[f"x{k:02}"] for k in range(2, 2+self.mags_per_ring)]).reshape((self.mags_per_ring,1)).flatten()
        
        #print(magnet_displacement)


        individual_rot_z=np.array([x[f"x{k:02}"] for k in range(0, 
                                                                self.mags_per_ring)]).reshape((self.mags_per_ring,1)).flatten()
        
        #individual_rot_y=np.array([x[f"x{k:02}"] for k in range(self.mags_per_ring*2, 
        #                                                         self.mags_per_ring*3)]).reshape((self.mags_per_ring,1)).flatten()
        
        # individual_rot_x=np.array([x[f"x{k:02}"] for k in range(5+self.mags_per_ring*3, 
        #                                                         5+self.mags_per_ring*4)]).reshape((self.mags_per_ring,1)).flatten()
        
        binary_placement_each_mag=np.array([x[f"x{k:02}"] for k in range(self.mags_per_ring*2, 
                                                                          self.mags_per_ring*(self.nrings+2))]).reshape((self.mags_per_ring*self.nrings,1)).flatten()#print(binary_placement_each_mag)

       
        # initialize and set sensor locations 
        col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_sphere(20,50,[0,0,0])
        sensor1 = magpy.Sensor(position=self.senrsor_points,style_size=2)

        col_sensors.add(sensor1)
        mag_vect = [1270,0,0]
        magnets = magpy.Collection(style_label='magnets')
        
        mintheta = 360/self.mags_per_ring
        theta = np.linspace(0,360-mintheta,self.mags_per_ring)*np.pi/180 #+ thetaoffset
        
        # theta = np.linspace(0,360-mintheta,self.mags_per_ring)*np.pi/180
        
        point_list=[]; # initialize list for ledger 
        
        bb=0
        for indx, jj in enumerate(zrows): # loop against each ring 
            aa=0
            #print(jj)
            #print(indx)
            for ii in range(theta.shape[0]): # loop for each theta in a ring 
                xx= rmin*np.cos(theta[ii]) # x position of 1 magnet 
                yy= rmin*np.sin(theta[ii]) # y position of 1 magnet 
                # rotx = individual_rot_x[aa]*180/np.pi # rotation along x each magnet block 
                # roty = individual_rot_y[aa]*180/np.pi # rotation along y each magnet block 
                rotx = 0 # rotation along x each magnet block 
                roty = 0 # rotation along y each magnet block 
                rotz = theta[ii]*180/np.pi + individual_rot_z[aa]*180/np.pi# rotation along z each magnet block 
                
                lengthmag = self.cube_side_length # length of magnet 
                if(binary_placement_each_mag[bb]==True): # if we want to place the magnet 
                    cube = magpy.magnet.Cuboid(magnetization=mag_vect,dimension=(self.cube_side_length,
                                                                             self.cube_side_length,lengthmag)) # make a cube magnet 
                    magnet_displacement = 0
                    #cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj*
                    #               (self.cube_side_length*np.sqrt(3)+1)) # get cube position 
                    cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj)
                    #cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj) #*
                    #               (self.cube_side_length*np.sqrt(3)+1)) # get cube position 
                    #print (magnet_displacement)
                    # print((x[0],y[0],lengthmag/2+magnet_displacement[aa]+jj*(self.cube_side_length*np.sqrt(3)+1)))
                    #cube.position=(1,1,1)
                    cube.xi = 0.04 # µr=1 +cube.xi
                    cube.orientation = R.from_euler('zyx',[rotz,roty,rotx],degrees=True) # rotation of cube 

                    magnets.add(cube) # add cube to magnet list 
                    point_list.append([xx,yy,lengthmag/2+magnet_displacement+jj,rotx,roty,rotz,0,0,0,0,0,self.cube_side_length,0])
                    tmp_df = pd.DataFrame(point_list,columns =['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot',
                                                       'Z-rot','Searched','CostValue','Used',
                                                       'Placement_index','Bmag','Magnet_length','Tag'])
            
            
                
                
                    cols_to_round = ['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched',
                             'CostValue','Used','Placement_index','Bmag','Magnet_length']
                
                    tmp_df[cols_to_round] = tmp_df[cols_to_round].round(3)
                    bb = bb+1
                    #print(bb)
                else:
                    bb = bb+1
                    #print(bb)
                
                aa=aa+1

            
        if len(magnets) == 0:
            B = np.zeros((21,3))
            BB = B + self.B_background
            print ('length of magnet is 0')
            eta = 1e6
            meanB0=0
        else:
            B = col_sensors.getB(magnets) # get B field of each magnet 
            tmpB = self.B_background - np.mean(self.B_background,axis=0)
            BB = tmpB + B
            eta = 42580*np.std(BB[:,0])
            meanB0 = np.mean(self.B_background[:,0] + B[:,0])
            #computing cost function for only one magnet
        # print('this is a test222:' +str(np.mean(B[:,0])))
        # print('this is a test333:' +str(np.mean(self.B_background[:,0])))

        #BB = B + self.B_background # get field of shim + background field 
        #if len(magnets) == 0:
            #eta = 1e6
            #meanB0=0
        #else:
            #eta, meanB0 = magsimulator.cost_func(BB,0)   #       
            

        print('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4)))

        f1=eta

        out["F"] = f1
        # out["G"] =len(magnets)-self.maxnumberofmagnets

      
    def get_mags(self, x):
        # SAS assign same definitions as in init
        #thetaoffset = np.array(x[f"x01"]).reshape((1,1))
        #thetaoffset=thetaoffset[0]
        # self.rmin = np.array(x[f"x01"]).reshape((1,1))
        rmin = self.rmin
        
        #zrows = np.array(x["x01"]).reshape((1,1))
        #zrows=int(zrows[0])
        zrows = self.zpos
        #print(zrows)
        #magnet_displacement=np.array([x[f"x{k:02}"] for k in range(2, 2+self.mags_per_ring)]).reshape((self.mags_per_ring,1)).flatten()
        
        #print(magnet_displacement)


        individual_rot_z=np.array([x[f"x{k:02}"] for k in range(0, 
                                                                self.mags_per_ring)]).reshape((self.mags_per_ring,1)).flatten()
        
        #individual_rot_y=np.array([x[f"x{k:02}"] for k in range(self.mags_per_ring*2, 
        #                                                         self.mags_per_ring*3)]).reshape((self.mags_per_ring,1)).flatten()
        
        # individual_rot_x=np.array([x[f"x{k:02}"] for k in range(5+self.mags_per_ring*3, 
        #                                                         5+self.mags_per_ring*4)]).reshape((self.mags_per_ring,1)).flatten()
        
        binary_placement_each_mag=np.array([x[f"x{k:02}"] for k in range(self.mags_per_ring*2, 
                                                                          self.mags_per_ring*(self.nrings+2))]).reshape((self.mags_per_ring*self.nrings,1)).flatten()#print(binary_placement_each_mag)

       
        # initialize and set sensor locations 
        col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_sphere(20,50,[0,0,0])
        sensor1 = magpy.Sensor(position=self.senrsor_points,style_size=2)

        col_sensors.add(sensor1)
        mag_vect = [1270,0,0]
        magnets = magpy.Collection(style_label='magnets')
        
        mintheta = 360/self.mags_per_ring
        theta = np.linspace(0,360-mintheta,self.mags_per_ring)*np.pi/180 #+ thetaoffset
        
        # theta = np.linspace(0,360-mintheta,self.mags_per_ring)*np.pi/180
        
        point_list=[]; # initialize list for ledger 
        
        bb=0
        for indx, jj in enumerate(zrows): # loop against each ring 
            aa=0
            #print(jj)
            #print(indx)
            for ii in range(theta.shape[0]): # loop for each theta in a ring 
                xx= rmin*np.cos(theta[ii]) # x position of 1 magnet 
                yy= rmin*np.sin(theta[ii]) # y position of 1 magnet 
                # rotx = individual_rot_x[aa]*180/np.pi # rotation along x each magnet block 
                # roty = individual_rot_y[aa]*180/np.pi # rotation along y each magnet block 
                rotx = 0 # rotation along x each magnet block 
                roty = 0 # rotation along y each magnet block 
                rotz = theta[ii]*180/np.pi + individual_rot_z[aa]*180/np.pi# rotation along z each magnet block 
                
                lengthmag = self.cube_side_length # length of magnet 
                if(binary_placement_each_mag[bb]==True): # if we want to place the magnet 
                    cube = magpy.magnet.Cuboid(magnetization=mag_vect,dimension=(self.cube_side_length,
                                                                             self.cube_side_length,lengthmag)) # make a cube magnet 
                    magnet_displacement = 0
                    #cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj*
                    #               (self.cube_side_length*np.sqrt(3)+1)) # get cube position 
                    cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj)
                    #cube.position= (xx,yy,lengthmag/2+magnet_displacement+jj) #*
                    #               (self.cube_side_length*np.sqrt(3)+1)) # get cube position 
                    #print (magnet_displacement)
                    # print((x[0],y[0],lengthmag/2+magnet_displacement[aa]+jj*(self.cube_side_length*np.sqrt(3)+1)))
                    #cube.position=(1,1,1)
                    cube.xi = 0.04 # µr=1 +cube.xi
                    cube.orientation = R.from_euler('zyx',[rotz,roty,rotx],degrees=True) # rotation of cube 

                    magnets.add(cube) # add cube to magnet list 
                    point_list.append([xx,yy,lengthmag/2+magnet_displacement+jj,rotx,roty,rotz,0,0,0,0,0,self.cube_side_length,0])
                    tmp_df = pd.DataFrame(point_list,columns =['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot',
                                                       'Z-rot','Searched','CostValue','Used',
                                                       'Placement_index','Bmag','Magnet_length','Tag'])
            
            
                
                
                    cols_to_round = ['X-pos', 'Y-pos', 'Z-pos','X-rot','Y-rot','Z-rot','Searched',
                             'CostValue','Used','Placement_index','Bmag','Magnet_length']
                
                    tmp_df[cols_to_round] = tmp_df[cols_to_round].round(3)
                    bb = bb+1
                    #print(bb)
                else:
                    bb = bb+1
                    #print(bb)
                
                aa=aa+1

            
        if len(magnets) == 0:
            B = np.zeros((21,3))
            BB = B + self.B_background
            print ('length of magnet is 0')
            eta = 1e6
            meanB0=0
        else:
            B = col_sensors.getB(magnets) # get B field of each magnet 
            tmpB = self.B_background - np.mean(self.B_background,axis=0)
            BB = tmpB + B
            eta = 42580*np.std(BB[:,0])
            meanB0 = np.mean(self.B_background[:,0] + B[:,0])
            #computing cost function for only one magnet
        # print('this is a test222:' +str(np.mean(B[:,0])))
        # print('this is a test333:' +str(np.mean(self.B_background[:,0])))

        #BB = B + self.B_background # get field of shim + background field 
        #if len(magnets) == 0:
            #eta = 1e6
            #meanB0=0
        #else:
            #eta, meanB0 = magsimulator.cost_func(BB,0)   #       
                        

        print('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4) ))


    
            
        return magnets, tmp_df, self.B_background, B, self.senrsor_points 

problem = MultiObjectiveMixedVariableProblem()

algorithm = MixedVariableGA(pop_size=100, survival=RankAndCrowdingSurvival())
# algorithm = NSGA2(pop_size=250,sampling=MixedVariableSampling(),                 mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),                  eliminate_duplicates=MixedVariableDuplicateElimination(),)
#algorithm = Optuna()

res = minimize(problem,
               algorithm,
               ('n_gen', 3000),
               verbose=True)



#%%
mag_vect = [1270,0,0]

filename1 = '../../field_mapping/map4_pos.mat'
filename2 = '../../field_mapping/map4_field.mat'
        #file = open(filename[:-4]+'pkl', 'rb')
tmp = magsimulator.data_from_matlab(filename2)
B_tmp =  tmp['field_matrix']  # main B0 field over which we are shimming !
B_field = B_tmp[:,0:3]   
# tmp = magsimulator.data_from_matlab(filename1)
tmp = magsimulator.data_from_matlab(filename1)
senrsor_points = tmp['pos'] 
magnets = magpy.Collection(style_label='magnets')
col_sensors = magpy.Collection(style_label='sensors')
#sensor1 = magsimulator.define_sensor_points_on_sphere(20,50,[0,0,0])
sensor1 = magpy.Sensor(position=senrsor_points,style_size=2)
col_sensors.add(sensor1)
#magnets.position = np.asarray([0,0,0]).T
xx=res.X
magnets,ledger, B_back, B_shim,Sens_pos = problem.get_mags(xx)

B = col_sensors.getB(magnets) # get B field of each magnet 
tmpB = B_back - np.mean(B_back,axis=0)
BB = tmpB + B
eta = 42580*np.std(BB[:,0])
meanB0 = np.mean(B_back[:,0] + B[:,0])

magsimulator.plot_magnets3D(ledger)
filename= 'passive_shim_eta_' + str(np.round(eta,4)) +'.xlsx'
ledger.to_excel(filename, index=False)

#%%
# magcadexporter.export_magnet_ledger_to_cad(filename,[0.2,0.2,0.2])

