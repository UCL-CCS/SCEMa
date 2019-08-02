# DeaLAMMPS

<img src="https://mvassaux.github.io/static/hmm_bicomposite_lo.jpg" align="right" width="50%" /> 

Heterogeneous Multiscale Method implementation featuring Deal.II (FE) and LAMMPS (MD). Enables simulations coupling semi-concurrently the evolution of an atomistic and a continuum system. The evolution of the continuum system drives the mechanical evolution of the periodic homogeneous atomistic replicas.

More details about this algorithm can be found in the following publication:
> Maxime Vassaux, Robin Richardson and Peter Coveney. [*The heterogeneous multiscale method applied to inelastic polymer mechanics.*](https://www.researchgate.net/publication/328930018_The_heterogeneous_multiscale_method_applied_to_inelastic_polymer_mechanics) Philosophical Transactions A, 377(2142), doi:10.1098/rsta.2018.0150.


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

The number of MD simulations can be drastically reduced through a graph reduction method with thresholding of "similarity" of one microstate's material history (strain vs time) with another - currently using L2 norm.

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

Publications:
> Vassaux, M., Richardson, R. A., & Coveney, P. V. (2019). The heterogeneous multiscale method applied to inelastic polymer mechanics. Philosophical Transactions of the Royal Society A, 377(2142), 20180150.

> Vassaux, M., Sinclair, R. C., Richardson, R. A., Suter, J. L., & Coveney, P. V. (2019). The Role of Graphene in Enhancing the Material Properties of Thermosetting Polymers. Advanced Theory and Simulations, 2(5), 1800168.

> Vassaux, M., Sinclair, R. C., Richardson, R. A., Suter, J. L., & Coveney, P. V. (2019). Towards high fidelity materials property prediction from multiscale modelling and simulation", Advanced Theory and Simulations. Accepted for publication.
