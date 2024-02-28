# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:01:59 2023

@author: la506
"""

import cadquery as cq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# takes the name of the excel ledger filename and the CAD template filename. It is important that the center of mass of the CAD object is centered at 0!

def export_magnet_ledger_to_cad(xlsfilename,templatefile):
    assy = cq.Assembly()
    
    
    ledger = pd.read_excel(xlsfilename, 
               dtype={'X-pos': float, 
                      'Y-pos': float, 
                      'Z-pos': float, 
                      'X-rot': float, 
                      'Y-rot': float,
                      'Z-rot': float,
                      'Searched': float,
                      'CostValue': float,
                      'Used': float,
                      'Placement_index': float,
                      'Bmag': float,
                      'Magnet_length': float,
                      'Tag': str})   
    
    ledger.columns = ['X-pos','Y-pos','Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length','Tag']
    ledger = ledger.reset_index(drop=True) #needed correction since rows are dropped

    for idx_ledger, ledger_row in ledger.iterrows():
        print('iterating index: ' + str(idx_ledger) + 'out of: ' + str(ledger.shape[0]))
        # cube_w_margine = (ledger_row['Magnet_length'],
        #                   ledger_row['Magnet_length'],
        #                   ledger_row['Magnet_length'])
        # if(idx_ledger==0):
        # result = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))

# rotating in the Z, Y, and X directions in this order. doing this at the center and then translating to a position


# LA commented out 9.20.23---------------------------
        # result = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        
        result = cq.importers.importStep(templatefile).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        # else:
        #     result2 = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        #     result = result.union(result2)
            
        
    # cq.exporters.export(result, filename[:-4]+ 'step')
        assy.add(result,color=cq.Color("red"))

    assy.save(xlsfilename[:-4]+ 'step')
     

