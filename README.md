# DeaLAMMPS
HMM implementation featuring Deal.II (FE) and LAMMPS (MD).

Works (at least) Deal.II/8.4.1 or above, and LAMMPS/17Nov16 compiled with RIGID package.

## Summary of work:

### Setting up the Finite Element simulation
Includes solving quasi-static or dynamic equilibrium of continuum mechanics, solve equilibrium incrementally, generate or import a mesh from gmsh, assign heterogenous materials properties.

### Relate strain to stress
Includes linear relation, MD simulation based relation, or statistically infered relation

### FE/MD Coupling
Includes passing down a macroscale strain, and transfering up an homogenized stress and/or stiffness tensor

### MD jobs scheduler
Includes splitting the processors adequately in between MD jobs, and Pilotjob.

### Database of mechanical states
Includes storing stress/strain space trajectories
