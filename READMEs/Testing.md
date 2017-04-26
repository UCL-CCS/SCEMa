# Standard virtual testing box

## Initial status:

The input files for the LAMMPS simulation of a reference box of polymer consists of one file (to be run once intially) that minimize the free energy of the generated box, heatup and finally cooldown the box content.

A second file is available, that should be called by DeaLAMMPS at every stress update on every quadrature point. The file applies an axial strain to an existing box initialized from the end of the initialization simulation.

## Current status:

The second file applies the given strain (fix deform) to an existing sample box and returns the homogenized stress (compute pressure).

## Future work:

The box should be restored from the binary data recorded either at the end of the initialization simulation or at the end of the previous stress update from that exact quadrature point.

The definition of the given strain and the choice of the restored data should be left to be set by the main C++ wrapper (dealammps.cpp).
