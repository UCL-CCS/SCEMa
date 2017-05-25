# Linear elastic domain

## Initial status:

The tangent stiffness tensor of the system is computed at the initialization stage and at every quadrature point state update.

## Current status:

Starting...

## Future work:

A first step during the initialization stage is to determine the limits of the linear elastic regime where we can assume that the tangent stiffness tensor remains constant, therefore we do not have to update run a quadrature point state update. This can be achieved, by applying finite strains instead of inifinitesimal ones during the 6 tests (one for each strain component) required to compute the stiffness tensor. When the applied strain amplitude induces a variation of the stiffness coefficients determined by the test, the amplitude is assumed to be the limit of linear elastic domain in that direction.

To test if a strain state is in the linear elastic regime, the material must never have left the elastic domain, and the strain should be lower than the amplitude limit in every of the 6 direction of loading.
