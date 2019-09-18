It is assumed here that:

a. DeaLAMMPS has been pulled here to a given directory named `/path/to/DeaLAMMPS`.
b. the executable `dealammps` has been compiled in a build directory, as described in the [instalation tutorial](instalation.md)

1. Create the simulation directory and move to it: 

```
mkdir /path/to/simulation
cd build
```

2. Import the initialised material data to `/path/to/simulation/nanoscale_input/`

- LAMMPS binary restart file: init.mat_nrep.bin
- Dimensions of the replica file: init.mat_nrep.length
- Initial stresses in the replica file: init.mat_nrep.stress
- Initial stiffness of the replica file: init.mat_nrep.stiff
- Atomistic material properties file: mat_nrep.json

3. Create a symbolic link in the simulation directory to the executable: 

```
ln -s /path/to/DeaLAMMPS/build/dealammps
```

4. Prepare the DeaLAMMPS input configuration file (inputs_dealammps.json)

5. Run the HMM simulation: 

```
aprun -n N_nodes ./dealammps inputs_dealammps.json
```

- `aprun`: job submission system execution command (in the present case PBS on ARCHER)
- `N_nodes`: number of nodes assigned to the simulation
