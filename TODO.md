### Things to be done
* Sort things to be done (duh...)
* Sort the content of READMEs
* How to store the current complete state of the system (for restart)?
  - For each gauss point, of each cell: stress, updstrain
  - For each replica of each gauss point of each cell: the binary lammps file
    - Could be replaced by a dump file of the atoms position and velocity and bond connectivities
  - For each degree of freedom: displacement, velocity
* Separate initialization of MD
  - For each replica, of each material type produce:  inital box stiffness, initial box stress, initial box dimensions and the init binary file
* Insert decision workflow to choose between MD simulation and Kriging estimation
* Use Pilotjob as a scheduler
* Write documentation
* Pass common features from all forks to master
