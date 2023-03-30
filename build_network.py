import numpy as np
from bmtk.builder.networks import NetworkBuilder
import math
import random

output_dir='network'

#######################################################################
##################### Create the cells ################################
#######################################################################
print("\nCreating Cells")

# Build the main network
net = NetworkBuilder('LUT_TOY')

# Specify number of cells in each population #

numPGN  = 1
#numBladaff  = 1

# Create the nodes ----------------------------------------
net.add_nodes(N=numPGN, level='low',pop_name='PGN',model_type='biophysical',model_template='hoc:PGN',morphology='blank.swc')
#net.add_nodes(N=numBladaff, level='high',pop_name='Bladaff',model_type='point_process', model_template= 'nrn:IntFire1', morphology='NULL', dynamics_params = 'IntFire1_exc_1.json')

####################################################################################
########################## Build and save network ##################################
####################################################################################

print("\nBuilding network and saving to directory \"" + output_dir + "\"")
net.build()

net.save_nodes(output_dir=output_dir)
#net.save_edges(output_dir=output_dir)

print("Done")
