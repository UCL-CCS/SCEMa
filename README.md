# DeaLAMMPS

<img src="https://mvassaux.github.io/static/hmm_bicomposite_lo.jpg" align="right" width="50%" /> 

Heterogeneous Multiscale Method implementation featuring Deal.II (FE) and LAMMPS (MD). Enables simulations coupling semi-concurrently the evolution of an atomistic and a continuum system. The evolution of the continuum system drives the mechanical evolution of the periodic homogeneous atomistic replicas.

More details about this algorithm can be found in the following publication:
> Maxime Vassaux, Robin Richardson and Peter Coveney. [*The heterogeneous multiscale method applied to inelastic polymer mechanics.*](https://www.researchgate.net/publication/328930018_The_heterogeneous_multiscale_method_applied_to_inelastic_polymer_mechanics) Philosophical Transactions A (in press), doi:10.1098/rsta.2018.0150. 

## Dependencies:
Works (at least) using [Deal.II/8.4.1](https://dealii.kyomu.43-1.org/downloads/dealii-9.0.1.tar.gz) or above, and [LAMMPS/17Nov16](https://lammps.sandia.gov/tars/lammps-17Nov16.tar.gz).

LAMMPS need to be compiled as a shared library, with the RIGID and USER-REAXC packages:
```sh
cd /path/to/lammps-17Nov16/src
make yes-RIGID
make yes-USER-REAXC
make mode=shlib mpi
```

Deal.II need to be compiled with dependencies required to run the tutorial [step-18](https://www.dealii.org/8.4.1/doxygen/deal.II/step_18.html#ElasticProblemoutput_results), that is the following dependencies: MPI, PETSc, METIS, BOOST, HDF5, LAPACK, MUPARSER, NETCDF, ZLIB, and UMFPACK. Complete instructions can be found [here](https://www.dealii.org/9.0.0/readme.html), and more specified directions for the ARCHER supercomputer, [here](https://github.com/ARCHER-CSE/build-instructions/tree/master/deal.II/build-gnu-64-petsc).

Continuum mechanics equilibrium equations are solved on the basis of a linear elastic material. Non-linear stress/strain beahvior is captured running MD simulations of a sample of material subject to the continuum strain when needed. 

A database is populated with the stress/strain history computed using MD simulations. When sufficiently filled, the database is used to infer the induced stress given a current strain history. Such technique reduces rapidly and drastically the number of MD simulations to run.

## Compile and run:
After installing separately LAMMPS and Deal.II, and building your MD input lammps data file.
```sh
cd /path/to/DeaLAMMPS
cp CMakeLists/example_machine.CMakeLists.txt CMakeLists.txt
mkdir build

cmake ../
./dealammps inputs_hmm.json
```
Additionally, a FE mesh can be imported from a GMSH file, and most of the parameters of the simulation can be found in dealammps.cc

