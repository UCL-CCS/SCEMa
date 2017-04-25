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

The algorithm has a predefined constant stiffness tensor homogoeneous over the whole sample.

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
