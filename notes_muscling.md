The current project aims at coupling a continuum and a molecular model using MUSCLE3.

The closest workflow available from the MUSCLE3 documentation is the `reaction_diffusion_mc_cpp.sh` example.

The workflow couples four CPP executables:
 * `mc_driver.cpp`: 
 	* this scripts generate a single set of configurations to simulate with the reaction/diffusion model, continuum.cc will do that once per continuum timestep.	
 * `diffusion.cpp`: 
 	* this is the model that is being simulated for multiple configurations of parameters in parallel.
 	* it receives its configuration from `load_balancer.cpp`, and sends back its results to `load_balancer.cpp`.
 * `load_balancer.cpp`:
 	* it receives multiple configurations of the diffusion model to simulate, splits them and sends them to `diffusion.cpp`
 	* it receives independent results from the multiple executions of `diffusion.cpp`, collates them and sends them to `mc_driver.cpp`.

Our adaptation of the `reaction_diffusion_mc_cpp.sh` example will contain three executables only:
 * `continuum.cc`:
 	* it is an iterative version of `mc_driver.cpp`.
 	* it sends to `load_balancer.cpp` the collection of configurations to simulate once per timestep.
 	* the collection of configurations to should be sent as a `ScaleBridgingData` structure, if possible.
 * `molecular.cc`:
 	* it is the exact of equivalent of `diffusion.cpp`, except that it does not send any information to a lower scale model (here, `reaction.cpp`/`reaction_mpi.cpp`), it should only receive and send back information to `load_balancer.cpp`.
 	* `molecular.cc` should feed on the `STMDProblem`class.
 * `load_balancer.cpp`:
 	* at the begining of each step, it should receive at each timestep a `ScaleBridgingData` structure and split it as multiple `MDSim` class objects.
 	* then before the end of the step it should receive multiple updated `MDSim` class objects and send an updated `ScaleBridgingData` structure.
 	* `load_balancer.cpp` has to complete two receive/send operations per timestep.
 	* `load_balancer.cpp` should feed on the `STMDSync` class.
 	* Functionalities to transfer from the FEProblem to the load balancer: 
		1. choice between MD or constitutive law derived strain
			1.1 strain threshold (1.e-10) below which constitutive law is enforced
		2. management and comparison of strain histories and subsequent clustering
		3. surrogate modelling
		4. load balancing of MD simulations
		5. average initial data from molecular system (e.g. stiffness)
