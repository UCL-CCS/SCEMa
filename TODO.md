### Things to be done
* Sort things to be done (duh...)
* Sort the content of READMEs
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
* Write documentation
* Pass common features from all forks to master
* Reflect on more than two scales
* Gather inputs in a .json file (parameters, path to: replicas, mesh, microstructure)
