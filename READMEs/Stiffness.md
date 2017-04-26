# Stiffness computation

## Initial status:

The algorithm has a predefined constant stiffness tensor homogeneous over the whole sample.

## Current status:

The update_quadrature_point function now stores the stresses and strains of the previous
iteration, and compute the strains (from the displacements) and stresses (calling LAMMPS
using the current strains).

The computation of a stress/strain state dependent stiffness tensor is implemented either in a
linear affine or secant fashion.

## Future work:

For the computation of the secant stiffness tensor, a choice must be made regarding the old
stress/strain chosen: either the last iteration state (k-1), or the last converged step (t-1).

The strain state should be formatted to be readable by a LAMMPS instance: a 6-elements vector (xx,yy,zz,xy,xz,yz). The
LAMMPS instance returns the stress state, which should be formatted back to a similar format as
the strain state.
