from pdbfixer import PDBFixer
from openmm.app import PDBFile


fixer = PDBFixer(filename='1UAO.pdb')
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
fixer.removeHeterogens(keepWater=False)
PDBFile.writeFile(fixer.topology, fixer.positions, open('input.pdb', 'w'))


from openmm import app, unit
from openmmtools import states, mcmc, multistate, cache
import openmm as mm
import numpy as np
import os, sys


pdbFile = app.PDBFile('input.pdb')
topology, positions = pdbFile.topology, pdbFile.positions


forceField = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
system = forceField.createSystem(
    topology=topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    hydrogenMass=4.0*unit.amu  # Set hydrogen mass to 4 Dalton for hydrogens bonded to heavy atoms 
)


n_replicas = 5  
min_temp = 350 * unit.kelvin
max_temp = 500 * unit.kelvin
simulation_time = 200 * unit.nanoseconds
timestep = 2.0 * unit.femtoseconds
steps_per_iteration = 500


platform = mm.Platform.getPlatformByName('CUDA')
properties = {
    'DeviceIndex': '0,1,2,3,4',  
    'Precision': 'mixed',         
    'UseCpuPme': 'false',        
    'Threads': '1'               
}
cache.global_context_cache.platform = platform
cache.global_context_cache.platform_properties = properties


reference_state = states.ThermodynamicState(system=system, temperature=min_temp)
sampler_state = states.SamplerState(positions)

#move = mcmc.GHMCMove(timestep=timestep, n_steps=steps_per_iteration)
move = mcmc.LangevinDynamicsMove(
    timestep=timestep,
    n_steps=steps_per_iteration,
    collision_rate=0.1 / unit.picoseconds 
)

simulation = multistate.ParallelTemperingSampler(
    mcmc_moves=move,
    number_of_iterations=int(simulation_time/(timestep*steps_per_iteration)),
    replica_mixing_scheme='swap-neighbors'
)


storage_path = 'parallel_tempering_output.nc'

num_atoms = topology.getNumAtoms()
reporter = multistate.MultiStateReporter(
    storage_path,
    checkpoint_interval=1
)

simulation.create(
    thermodynamic_state=reference_state,
    sampler_states=sampler_state,
    storage=reporter,
    min_temperature = min_temp,
    max_temperature = max_temp,
    n_temperatures=n_replicas
)


print("Minimizing energy across all replicas...")
simulation.minimize()


print(f"Running {simulation_time} simulation with exchange attempts every {timestep*steps_per_iteration} ")
simulation.run()
