### Things to be done
* Sort the content of READMEs
* Setup a docker image of DeaLAMMPS based on existing dealii existing [ones](https://hub.docker.com/r/dealii/dealii/)
* Profile Deal.ii + I/O 
* Write a Python wrapper around the c++ functions found in dealammps.cc or single_md.cc compiled as a shared library
  - the Python wrapper would only execute the higher-level functions such as `do_timestep` (and the other ones in the HMMProblem class)
  - complete separation of the different models and synchronisation relying on MUSCLE
    - base the work on pre-separated models in version *standalone_md*
  - see the following [slides](https://figshare.com/articles/Interfacing_Python_to_C_UCL_20_June_2018/6626639) for tutorials on interface Python and C++
* Outputs management
  - store for each iteration (timestep), the bare minimum of information to restart the simulation (see restarting process)
  - prepare executable to output visualization files (VTK/XML) from restart checkpoints, and lammps scripts to compute time averaged variables (either global or local)
  - build database (SQL, PostgreSQL, ...) from checkpoints or visualization files
* Improve restarting process
  - serialize global simulation state: continuum (triangulation, nodes data, cell data), atomistic (atoms position, topology)
  - finite element cell number independence (location-based? else?)
* Improve homogenization procedure
  - enhanced stress homogenization, stiffness from fluctuations (Luding, S.)
  - MercuryDPM (Luding, S.), LIMEpy (Leither, K., ARL)
* Adaptative mesh refinement
  - based on deal.ii capabilities
  - transfer mother cell features (strain, atomic model) to child cells
  - handle cell renumbering
* Avoid one `fe_problem.h` per FE configuration (mesh+BC)
* Separate strain checking and spline comparison from the `FE_Problem` class
* Random combination of MD replicas associated to each FE quadrature point (or at least a different initial velocity)
* Upscale local information from MD simulation to the FE problem for spatial visualization
  - bond count
  - dissipated energy via thermoset (but hard to compute)
* Write documentation
* Pass common features from all forks to master
* N-scales extension
   - apply periodic boundary conditions on the mesoscale finite element model

