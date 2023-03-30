"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import numpy as np
import os
import sys
from bmtk.simulator import bionet
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.analyzer.spike_trains import plot_raster
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.pointprocesscell import PointProcessCell
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.simulator.bionet.io_tools import io

from neuron import h


pc = h.ParallelContext()


class FeedBackLoop(SimulatorMod):
    def __init__(self):
        self._spike_events = {}
        self._synapses = {}
        self._netcons = {}
        self._spike_records = {}
        self._vect_stims = {}
        self._spikes = {}

        self._block_length_ms = 0.0
        self._n_cells = 0
        #self._current_input_rate = 5.0
        
        self.blad_fr = 1.0
        self._prev_glob_press = 0.0
        self._glob_press = 0.0 
        self.times = []
        self.b_vols = []
        self.b_pres = []

    def _set_spike_detector(self, sim):
        for gid, cell in sim.net.get_local_cells().items():
            tvec = sim._spikes[gid]
            self._spike_records[gid] = tvec

    def initialize(self, sim):
        self._block_length_ms = sim.nsteps_block*sim.dt
        self._n_cells = len(sim.net.get_local_cells())
        print(sim.net.get_local_cells())

        self._spikes = h.Vector()  # start off with empty input
        vec_stim = h.VecStim()
        vec_stim.play(self._spikes)
        self._vect_stim = vec_stim

        for gid, cell in sim.net.get_local_cells().items():
            self._spike_events[gid] = np.array([])
            # For each cell we setup a network connection, NetCon object, that stimulates input as a series of spike
            # events mimicking a synapse. For this simple example each cell recieves only 1 virtual synapse/netcon.
            # To have more than 1 netcon in each cell you can add an extra internal loop, and _synapses and _netcons
            # will be a dictionary of lists
            if isinstance(cell, BioCell):
                # For biophysicaly detailed cells we use an Synapse object that is placed at the soma. If you want to
                # place it at somewhere different than the soma you can use the following code:
                #   seg_x, sec_obj = cell.morphology.find_sections(
                #       sections_names=[axon, soma, dend, apic],
                #       distance_ranges=[0.0, 1000.0]
                #   )
                #   syn = h.Exp2Syn(seg_x, sec=sec_obj
                syn = h.Exp2Syn(0.5, sec=cell.hobj.soma[0])
                syn.e = 0.0
                syn.tau1 = 0.1
                syn.tau2 = 0.3
                self._synapses[gid] = syn

                # create a NetCon connection on the synpase using the array of spike-time values
                nc = h.NetCon(vec_stim, syn)
                nc.threshold = sim.net.spike_threshold
                nc.weight[0] = 0.2
                nc.delay = 1.0
                self._netcons[gid] = nc

            elif isinstance(cell, PointProcessCell):
                nc = h.NetCon(vec_stim, cell.hobj)
                self._netcons[gid] = nc
                nc.weight[0] = 15 #10.2
                nc.delay = .05

        self._set_spike_detector(sim)
        pc.barrier()

    def step(self, sim, tstep):
        pass
    
    def block(self, sim, block_interval):
        block_length = sim.nsteps_block*sim.dt/1000.0
        t = sim.h.t-block_length*1000.0
        
        #### BLADDER EQUATIONS ####    
    # Grill, et al. 2016
        def blad_vol(vol):
            f = 1.5*20*vol - 10 #1.5*20*vol-10
            return f

        # Grill function returning pressure in units of cm H20
	    # Grill, et al. 2016
        def pressure(fr,v):
            p = 0.2*fr + 1.0*v
            p = max(p,0.0)
            return p 

        # Grill function returning bladder afferent firing rate in units of Hz
	    # Grill, et al. 2016
        def blad_aff_fr(p):
            fr1 = -3.0E-08*p**5 + 1.0E-5*p**4 - 1.5E-03*p**3 + 7.9E-02*p**2 - 0.6*p
            fr1 = max(fr1,5.0)
            return fr1 # Using scaling factor of 5 here to get the correct firing rate range

    ### STEP 1: Calculate PGN Firing Rate ###
        print(f'Caclulating firing rates for times {block_interval[0]*sim.dt} to {block_interval[1]*sim.dt} ms')
        print('node_id\tHz')
        summed_fr = 0
        for gid, tvec in self._spike_records.items():
            # self._spike_records is a dictionary of the recorded spikes for each cell in the previous block of
            #  time. When self._set_spike_detector() is called it will reset/empty the spike times. If you want to
            #  print/save the actual spike-times you can call self._all_spikes[gid] += list(tvec)
            if gid == 0:
              n_spikes = len(tvec)
              fr = n_spikes / (self._block_length_ms/1000.0)
              summed_fr += fr
              print(f'{gid}\t\t{fr}')
        print(f'firing rate avg: {summed_fr / self._n_cells}')
        
        # Grill 
        PGN_fr = max(2.0E-03*fr**3 - 3.3E-02*fr**2 + 1.8*fr - 0.5, 0.0)
        print("Grill PGN fr = {0} Hz".format(PGN_fr))

    ### STEP 2: Volume Calculations ###
        v_init = 0.05       # TODO: get biological value for initial bladder volume
        fill = 0.05 	 	# ml/min (Asselt et al. 2017)
        fill /= (1000 * 60) # Scale from ml/min to ml/ms
        void = 4.6 	 		# ml/min (Streng et al. 2002)
        void /= (1000 * 60) # Scale from ml/min to ml/ms
        max_v = 1.5 		# ml (Grill et al. 2019) #0.76
        vol = v_init
        
        # Filling
        if t < 60000 and vol < max_v:
            vol = fill*t*20 + v_init
        # Voiding
        elif self.blad_fr > 10:
            vol = max_v - void*(60000-t)*100
        
        # Maintain minimum volume
        if vol < v_init:
            vol = v_init
        
        # Grill
        grill_vol = blad_vol(vol)
        
    ### STEP 3: Pressure and Bladder Afferent FR Calculations ###
        p = pressure(PGN_fr, grill_vol)
        self.blad_fr = blad_aff_fr(p)
        
    ### STEP 4: Update the input spikes each cell recieves in the next time block
        # Calculate the start and stop times for the next block
        next_block_tstart = block_interval[1]*sim.dt
        next_block_tstop = next_block_tstart+self._block_length_ms

        # For this simple example we just create a randomized series of spike for the next time block for each of the
        #  14 cells. The stimuli input rate (self._current_input_rate) is increamented by 10 Hz each block, for more
        #  realistic simulations you can use the firing-rates calcualted above to adjust the incoming stimuli.
        print("Calculated Bladder Afferent Firing Rate: {0}".format(self.blad_fr))
        psg = PoissonSpikeGenerator()
        psg.add(
            node_ids= 0,
            firing_rate= self.blad_fr,
            times=(next_block_tstart/1000.0 + 0.01, next_block_tstop/1000.0),
            population= 'PGN',
        )
        
        psg.add_spikes([0], [next_block_tstop], population = "PGN")
        psg.to_csv("spikes.csv")
        #self._current_input_rate += 10.0

        for gid, cell in sim.net.get_local_cells().items():
            spikes = psg.get_times(gid, population='PGN')
            spikes = np.sort(spikes)
            #spikes = np.arange(next_block_tstart/1000.0 + 0.1, next_block_tstop/1000.0, 0.1).tolist()
            print("HEllo: \n {0}".format(spikes))
            if len(spikes) == 0:
                continue

            # The next block of code is where we update the incoming/virtual spike trains for each cell, by adding
            # each spike to the cell's netcon (eg synapse). The only caveats is the spike-trains array must
            #  1. Have atleast one spike
            #  2. Be sorted
            #  3. first spike must occur after the delay.
            # Otherwise an error will be thrown.
            self._spike_events[gid] = np.concatenate((self._spike_events[gid], spikes))
            nc = self._netcons[gid]
            for t in spikes:
                nc.event(t)

        self._set_spike_detector(sim)
        pc.barrier()
        
    ### STEP 5: Save Calculations ####
        self._prev_glob_press = self._glob_press
        self._glob_press = p 

        io.log_info('PGN firing rate = %.2f Hz' %fr)
        io.log_info('Volume = %.2f ml' %vol)
        io.log_info('Pressure = %.2f cm H20' %p)
        io.log_info('Bladder afferent firing rate = {:.2f} Hz'.format(self.blad_fr))

        # Save values in appropriate lists
        self.times.append(t)
        self.b_vols.append(vol)
        self.b_pres.append(p)

    def finalize(self, sim):
        pass
        