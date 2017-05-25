# Stiffness computation

## Initial status:

The algorithm has a predefined constant stiffness tensor homogeneous over the whole sample.

## Current status:

The update_quadrature_point function now stores the stresses and strains of the previous iteration, and compute the strains (from the displacements) and stresses (calling LAMMPS using the current strains).

The computation of a microstate dependent stiffness tensor is computed in LAMMPS based on the reference ELASTIC example. Such routine computes the tangent stiffness at the current stress/strain state. The computation of the stiffness is done at the initiatialization of the HMM (in 'lammps_initiation()'). It is done as well for each quadrature point when a the new stress tensor is computed.

## Future work:

Introduce an elastic regime where stresses are not computed every time using LAMMPS, but checked only less frequently. It could also be possible to go back to this elastic regime if the algorithm observes that the behaviour does not change significantly for certain number of iteration. (see Linear elastic domain)

Outside of the elastic regime, the stiffness tensor should not be computed at every iteration. Maybe only at every time-step or after a sufficient variation of the stress/strain ratio (which is not the stiffness tensor, but some indicator of the stiffness in the testing direction).
