# Database of Microstates

## Initial status:

Every stress update at a quadrature point requires a new call to lammps, starting either from the initial sample and applying the complete strain, or starting from the stored last computed state (or atom positions or microstates) of the sample in that quadrature point.

The purpose of the database of is to avoid throwing lammps calls, when the stresses to be computed depend on a pair of initial state (or atom positions or microstates) and the applied strain, that have already been computed.

## Current status:

Not started...

## Future work:

In order to use data that has been computed for one quadrature point, for an other one, one has to know if the strain tensor applied and the initial sample state (or atom positions or microstates) are identical.

While one can compare two strain tensors, one might find difficult to compare two sample states.

Therefore, we have to define indicators that allow to compare two samples state. Ideally, if the indicators are sufficiently well defined, when the indacators are equal, the two samples states are identical.

These indicators would serve as metadata for our database of the samples states (or atom positions or microstates).

Once these indicators are defined, one might find interesting to implement extrapolation techniques (such as Kriging), to avoid throwing lammps call when the samples states or applied strains are not completely identical but quite close.
