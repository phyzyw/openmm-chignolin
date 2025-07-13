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

from openmm import app
from openmm import unit
import os
import openmm as mm
import numpy as np
import sys
pdbFile = app.PDBFile('input.pdb')
topology, positions = pdbFile.topology, pdbFile.positions
ntimes = 1
totalSteps = 200000000
for n in range(ntimes):
    outdir = 'outdir_'+str(n)
    os.makedirs(outdir, exist_ok=True)
    forceField = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    system = forceField.createSystem(
        topology=topology, 
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=4.0*unit.amu  # Set hydrogen mass to 4 Dalton for hydrogens bonded to heavy atoms 
    )
    integrator = mm.LangevinIntegrator(
        350*unit.kelvin, 
        0.1/unit.picoseconds,
        2.0*unit.femtoseconds
    )
    platform = mm.Platform.getPlatformByName('CUDA')
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    '''
    Geometry Optimization
    '''
    print(f'Trajectory {n}: Minimizing...')
    simulation.context.setVelocitiesToTemperature(350*unit.kelvin)  
    simulation.minimizeEnergy(maxIterations=10000)

    state = simulation.context.getState(getPositions=True)
    app.PDBFile.writeFile(topology, state.getPositions(), open(f'{outdir}/geoOpt.pdb', 'w'))
    
    '''
    NVT thermostat to equilibrium
    '''
    simulation.step(20000)
    
    '''
    NVT run,  save the frams per 1000 steps with the PDB format in outdir with the fileName--'traj.pdb'
    '''
    simulation.reporters.append(app.PDBReporter(f'{outdir}/traj.pdb', 1000))
    simulation.reporters.append(app.StateDataReporter(
        f'{outdir}/state.csv', 1000, step=True, potentialEnergy=True, temperature=True,
        progress=True, remainingTime=True, speed=True, totalSteps=totalSteps, separator=','
    ))
    simulation.reporters.append(app.CheckpointReporter(f'{outdir}/restart.chk', 500))
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, 2000, step=True, potentialEnergy=True, temperature=True,
        progress=True, remainingTime=True, speed=True, totalSteps=totalSteps, separator=','
    ))
    simulation.step(totalSteps)

    
