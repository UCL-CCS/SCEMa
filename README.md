# DeaLAMMPS
HMM implementation featuring Deal.II (FE) and LAMMPS (MD)

## Parallelization implementation
The purpose of this branch is to properly parallelize this algorithm using MPI and PETSc wrapper.

### Initial status:

The algorithm integrates the iterative solution algorithm introduced in step-15,
as well as the time-stepping algorithm introduced in step-18. The algorithm is sequential.

### Current status:

The algorithm includes the parallization features of introduced in step-17, but
mostly those in step-18.

### Future work:

The solution vectors (newton-update, incremental-displacement, and solution) and stresses
vectors (quadrature_point_history) are local vectors (namely not shared or distributed in
between the processes), this is not scalable in terms of memory usage. This must be looked
upon (see step-17 or step-40, or maybe elsewhere:
http://www.dealii.org/8.4.1/doxygen/deal.II/group__distributed.html).

The mesh is shared between processes (a complete copy of the mesh is available on
every process). Although the mesh construction is parallel, its storage costs important amounts
of memory. Once again this is not scalable and must be fixed (see step-40, using the library
p4est).

The generation of output visualization files (.vtk, .pvtu, .pvd) is a parallel
process (as explained in step-18). That might have to be checked according to step-40 it is not
or at least not completely.

Finally, some minor improvements could be done (see comments in code).

## Stiffness computation

### Initial status:

The algorithm has a predefined constant stiffness tensor homogeneous over the whole sample.

### Current status:

The update_quadrature_point function now stores the stresses and strains of the previous
iteration, and compute the strains (from the displacements) and stresses (calling LAMMPS
using the current strains).

The computation of a stress/strain state dependent stiffness tensor is implemented either in a
linear affine or secant fashion.

### Future work:

For the computation of the secant stiffness tensor, a choice must be made regarding the old
stress/strain chosen: either the last iteration state (k-1), or the last converged step (t-1).

The strain state should be formatted to be readable by a LAMMPS instance: a 6-elements vector (xx,yy,zz,xy,xz,yz). The
LAMMPS instance returns the stress state, which should be formatted back to a similar format as
the strain state.

## Standard virtual testing box

### Initial status:

The input files for the LAMMPS simulation of a reference box of polymer consists of one file (to be run once intially) that minimize the free energy of the generated box, heatup and finally cooldown the box content.

A second file is available, that should be called by DeaLAMMPS at every stress update on every quadrature point. The file applies an axial strain to an existing box initialized from the end of the initialization simulation.

### Current status:

The second file applies the given strain (fix deform) to an existing sample box and returns the homogenized stress (compute pressure).

### Future work:

The box should be restored from the binary data recorded either at the end of the initialization simulation or at the end of the previous stress update from that exact quadrature point.

The definition of the given strain and the choice of the restored data should be left to be set by the main C++ wrapper (dealammps.cpp).

## Database of Microstates

### Initial status:

Every stress update at a quadrature point requires a new call to lammps, starting either from the initial sample and applying the complete strain, or starting from the stored last computed state (or atom positions or microstates) of the sample in that quadrature point.

The purpose of the database of is to avoid throwing lammps calls, when the stresses to be computed depend on a pair of initial state (or atom positions or microstates) and the applied strain, that have already been computed.

### Current status:

Not started...

### Future work:

In order to use data that has been computed for one quadrature point, for an other one, one has to know if the strain tensor applied and the initial sample state (or atom positions or microstates) are identical.

While one can compare two strain tensors, one might find difficult to compare two sample states.

Therefore, we have to define indicators that allow to compare two samples state. Ideally, if the indicators are sufficiently well defined, when the indacators are equal, the two samples states are identical.

These indicators would serve as metadata for our database of the samples states (or atom positions or microstates).

Once these indicators are defined, one might find interesting to implement extrapolation techniques (such as Kriging), to avoid throwing lammps call when the samples states or applied strains are not completely identical but quite close.
