It is assumed here that:

a. DeaLAMMPS has been pulled here to a given directory named `/path/to/DeaLAMMPS`.
b. the executable init_material has been compiled in a build directory, as described in the [instalation tutorial](instalation.md)

1. You will require two input files for each new nanoscale material:
  - A LAMMPS data/structure file. Named with a material label and replica number (starting from 1), like so `mat_nrep.data`; e.g. `polymer_1.data`
  - Atomistic material properties file, `mat_nrep.json`.

2. Prepare the initialisation input configuration file `inputs_init_material.json`. A default is provided in the root DeaLAMMPS directory.

3. Prepare a directory to run the material initialization.

- `mkdir -p new_material/material_input`
- `cp inputs_init_material.json new_material/`
- `cp path/to/inputfiles new_material/material_input/`
- `cd new_material`
- `ln -s ../build/init_material`

4. Run the initialisation of the systems simulation: 

```
aprun -n N_nodes ./init_material inputs_init_material.json
```

- `aprun`: job submission system execution command (in the present case PBS on ARCHER)
- `N_nodes`: number of nodes assigned to the simulation

5. Retrieve the HMM simulation input files from the “nanoscale input” directory

- LAMMPS binary restart file: init.mat_nrep.bin
- Dimensions of the replica file: init.mat_nrep.length
- Initial stresses in the replica file:  init.mat_nrep.stress
- Initial stiffness of the replica file:  init.mat_nrep.stiff

