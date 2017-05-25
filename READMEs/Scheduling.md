# Linear elastic domain

## Initial status:

Running all the update tasks using LAMMPS at a given iteration is done in a synchronous fashion. The distribution of the update tasks is planned in advance (just before starting the updates), and every group of cores/processes has the same amount of update tasks to accomplish. This could lead to important idle times if some update tasks happen to be longer than others.

## Current status:

Starting...

## Future work:

Implement the QCG broker...
