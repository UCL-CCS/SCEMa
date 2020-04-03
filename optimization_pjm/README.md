# Routine for the optimisation of the scheduling of molecular dynamics simulations when using a PilotJob-Manager

The script `optimization_hmm.py` refers to results published in:

> Saad A. Alowayyed, Maxime Vassaux, Ben Czaja, Peter V. Coveney and Alfons G. Hoekstra. [*Towards Heterogeneous Multi-scale Computing on Large Scale Paralllel Supercomputers.*](https://superfri.org/superfri/article/view/281) Supercomputing Frontiers and Innovation, 377(2142), doi:10.14529/jsfi190402.

When using the PilotJob-Manager (PJM) scheduling system to assign suballocations (of the total allocation assigned to SCEMa) to each individual MD simulation, it is possible to assign a unique suballocation size.

The script uses walltimes of previous MD simulation (which depend on the strain applied, and the number of cores assigned), to predict the walltime of new MD simulations on one core. Then using a scaling law (i.e the speedup vs. the number of cores assigned to an MD simulation), the script assigns a given number of cores to each new MD simulation to perform so that each of these simulation lasts the same amount of walltime, independentely of the strain applied.

The data on which rely the script is acquired during previous benchmarking runs and has been stored in a database provided by the ComPat project.
