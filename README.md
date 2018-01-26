# DeaLAMMPS
Heterogenous Multiscale Method implementation featuring Deal.II (FE) and LAMMPS (MD). Works (at least) Deal.II/8.4.1 or above, and LAMMPS/17Nov16 compiled with RIGID package.

Continuum mechanics equilibrium equations are solved on the basis of a linear elastic material. Non-linear stress/strain beahvior is captured running MD simulations of a sample of material subject to the continuum strain when needed. 

A database is populated with the stress/strain history computed using MD simulations. When sufficiently filled, the database is used to infer the induced stress given a current strain history. Such technique reduces rapidly and drastically the number of MD simulations to run.

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
