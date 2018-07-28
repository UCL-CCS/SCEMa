### Things to be done
* Sort the content of READMEs
* Setup a docker image of DeaLAMMPS based on existing dealii existing [ones](https://hub.docker.com/r/dealii/dealii/)
* Profile Deal.ii + I/O 
* Write a Python wrapper around the c++ functions found in dealammps.cc or single_md.cc compiled as a shared library
  - the Python wrapper would only execute the higher-level functions such as `do_timestep` (and the other ones in the HMMProblem class)
  - Complete separation of the different models and synchronisation relying on MUSCLE
    - base the work on pre-separated models in version *standalone_md*
  - See the following [slides](https://figshare.com/articles/Interfacing_Python_to_C_UCL_20_June_2018/6626639) for tutorials on interface Python and C++
* Improve restarting process
  - continuity of timesteps
  - finite element cell number independence (location-based? else?)
* Improve homogenization procedure
  - Enhanced stress homogenization, stiffness from fluctuations (Luding, S.)
  - MercuryDPM (Luding, S.), LIMEpy (Leither, K., ARL)
* Adaptative mesh refinement
  - based on deal.ii capabilities
  - transfer mother cell features (strain, atomic model) to child cells
  - handle cell renumbering
* Separate initialization of MD
  - For each replica, of each material type produce:  inital box stiffness, initial box stress, initial box dimensions and the init binary file
* Use Pilotjob as a scheduler
  - use `split-into` option and assign value such as: $n_{split} = min \(n_{nodes, tot}, n_{microjobs}\)}$
  - assign `min` option proportional to applied strain 
* Avoid updating multiple cells that have the same mechanical history at the current time-step by fitting spline to the strain history, and measuring the distance in between
  - if two cells have the same history, run the atomic model for only one of them, and for the other copy the stress and the final state of the atomic model
  - later use GP regression, to relax the constraint on the similarity of the strain history down to "close enough".
  - insert decision workflow to choose between MD simulation and GPR regression
* Write documentation
* Pass common features from all forks to master
* N-scales extension
   - apply periodic boundary conditions on the mesoscale finite element model

