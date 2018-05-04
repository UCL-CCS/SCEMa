### Things to be done
* Sort things to be done (duh...)
* Sort the content of READMEs
* Write a Python wrapper around the c++ functions found in dealammps.cc or single_md.cc compiled as a shared library
  - the Python wrapper would only execute the higher-level functions such as `do_timestep` (and the other ones in the HMMProblem class)
* Improve restarting process
  - continuity of timesteps
  - finite element cell number independence (location-based? else?)
* Improve homogenization procedure
  - Enhanced stress homogenization, stiffness from fluctuations (Luding, S.)
  - MercuryDPM (Luding, S.), LIME (Leither, K., ARL)
* Separate initialization of MD
  - For each replica, of each material type produce:  inital box stiffness, initial box stress, initial box dimensions and the init binary file
* Complete separation of the different models and synchronisation relying on MUSCLE
  - base the work on pre-separated models in version *standalone_md*
* Insert decision workflow to choose between MD simulation and Kriging estimation
* Use Pilotjob as a scheduler
* Avoid updating multiple cells that have the same mechanical history at the current time-step by fitting spline to the strain history, and measuring the distance in between
  - if two cells have the same history, run the atomic model for only one of them, and for the other copy the stress and the final state of the atomic model
  - later use GP regression, to relax the constraint on the similarity of the strain history down to "close enough".
* Write documentation
* Pass common features from all forks to master
* Reflect on more than two scales
* Gather inputs in a .json file (parameters, path to: replicas, mesh, microstructure)
