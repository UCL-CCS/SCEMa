# Standard virtual testing box

## Initial status:

The input files for the LAMMPS simulation of a reference box of polymer consists of one file (to be run once intially) that minimize the free energy of the generated box, heatup and finally cooldown the box content. Using the ELASTIC example from LAMMPS documentation, the tangent stiffness of the testing box is also computed

A second file is available, that should be called by DeaLAMMPS at every stress update on every quadrature point. The file applies an axial strain to an existing box initialized from the end of the initialization simulation. Once again the stiffness is computed.

## Current status:

The second file applies the given strain (fix deform) to an existing sample box and returns the homogenized stress (compute pressure).

The initial state of the testing box is restored from the binary data recorded either at the end of the initialization simulation or at the end of the previous stress update from that exact quadrature point. The choice is given in the C++ wrapper.

The wrapper retrieves the compute stress as a vector and converts it to a standard square rank 2 tensor and the stiffness tensor as a square rank 4 tensor.

## Future work: 

Check the whole interplay of the two namespaces and write tests...
