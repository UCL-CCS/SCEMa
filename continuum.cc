/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

// TODO: Functionalities to transfer to the load balancer:
// 0. average initial data from molecular system (e.g. stiffness)
// 1. choice between MD or constitutive law derived strain
// 1.1 strain threshold (1.e-10) below which constitutive law is enforced
// 2. management and comparison of strain histories and subsequent clustering
// 3. surrogate modelling
// 4. load balancing of MD simulations

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>
#include <chrono>         // std::chrono::seconds, header file for wall-time measurement

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
//#include "boost/filesystem.hpp

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/mpi.h>

// Specifically built header files
#include "headers/read_write.h"
#include "headers/math_calc.h"
#include "headers/scale_bridging_data.h"

// Include of the FE model to solve in the simulation
#include "headers/FE_problem.h"

namespace CONT
{
	using namespace dealii;

	template <int dim>
	class CONTProblem
	{
	public:
		CONTProblem ();
		~CONTProblem ();
		void run (std::string inputfile);

	private:
		void read_inputs(std::string inputfile);
		void set_repositories ();
		void do_timestep ();
		void share_scale_bridging_data (ScaleBridgingData &scale_bridging_data);

		FEProblem<dim> 			*fe_problem = NULL;
		
		MPI_Comm 			world_communicator;
		const int 			n_world_processes;
		const int 			this_world_process;
		int 				world_pcolor;

		unsigned int			machine_ppn;

		ConditionalOStream 		hcout;

		int				start_timestep;
		int				end_timestep;
		double              		present_time;
		double              		fe_timestep_length;
		double              		end_time;
		int        			timestep;
		int        			newtonstep;

		int				fe_degree;
		int				quadrature_formula;
		std::string			twod_mesh_file;
                double                          extrude_length;
                int                             extrude_points;

		std::vector<std::string>	mdtype;
		unsigned int			nrepl;
		boost::property_tree::ptree	input_config;

		int				freq_checkpoint;
		int				freq_output_visu;
		int				freq_output_lhist;
		int				freq_output_lbcforce;
		
		std::string                 	macrostatelocin;
		std::string                	macrostatelocout;
		std::string			macrostatelocres;
		std::string			macrologloc;
		
	};



	template <int dim>
	CONTProblem<dim>::CONTProblem ()
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	CONTProblem<dim>::~CONTProblem ()
	{}




	template <int dim>
	void CONTProblem<dim>::read_inputs (std::string inputfile)
	{
	    std::ifstream jsonFile(inputfile);
	    try{
		    read_json(jsonFile, input_config);
	    }
	    catch (const boost::property_tree::json_parser::json_parser_error& e)
	    {
	        hcout << "Invalid JSON FE input file (" << inputfile << ")" << std::endl;  // Never gets here
	    }
            	    
	    boost::property_tree::read_json(inputfile, input_config);
            
	    // Continuum timestepping
	    fe_timestep_length 	= input_config.get<double>("continuum time.timestep length");
	    start_timestep 	= input_config.get<int>("continuum time.start timestep");
	    end_timestep 	= input_config.get<int>("continuum time.end timestep");

	    // Continuum meshing
	    fe_degree 		= input_config.get<int>("continuum mesh.fe degree");
	    quadrature_formula 	= input_config.get<int>("continuum mesh.quadrature formula");

	    // Continuum input, output, restart and log location
		macrostatelocin	 = input_config.get<std::string>("directory structure.macroscale input");
		macrostatelocout = input_config.get<std::string>("directory structure.macroscale output");
		macrostatelocres = input_config.get<std::string>("directory structure.macroscale restart");
		macrologloc 	 = input_config.get<std::string>("directory structure.macroscale log");


		// Molecular dynamics material data
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
				get_subbptree(input_config, "materials info").get_child("list of materials.")) {
			mdtype.push_back(v.second.data());
		}

		// Computational resources
		machine_ppn = input_config.get<unsigned int>("computational resources.machine cores per node");

		// Output and checkpointing frequencies
		freq_checkpoint   = input_config.get<int>("output data.checkpoint frequency");
		freq_output_lhist = input_config.get<int>("output data.analytics output frequency");
		freq_output_lbcforce = input_config.get<int>("output data.loaded boundary force output frequency");
		freq_output_visu  = input_config.get<int>("output data.visualisation output frequency");
		
		// Print a recap of all the parameters...
		hcout << "Parameters listing:" << std::endl;
		hcout << " - FE timestep duration: "<< fe_timestep_length << std::endl;
		hcout << " - Start timestep: "<< start_timestep << std::endl;
		hcout << " - End timestep: "<< end_timestep << std::endl;
		hcout << " - FE shape funciton degree: "<< fe_degree << std::endl;
		hcout << " - FE quadrature formula: "<< quadrature_formula << std::endl;
		hcout << " - List of material names: "<< std::flush;
		for(unsigned int imd=0; imd<mdtype.size(); imd++) 
		{
			hcout << " " << mdtype[imd] << std::flush; 
		}
		hcout << std::endl;

		
		hcout << " - Number of cores per node on the machine: "<< machine_ppn << std::endl;
		hcout << " - Frequency of checkpointing: "<< freq_checkpoint << std::endl;
		hcout << " - Frequency of writing FE data files: "<< freq_output_lhist << std::endl;
		hcout << " - Frequency of writing FE visualisation files: "<< freq_output_visu << std::endl;
		hcout << " - FE input directory: "<< macrostatelocin << std::endl;
		hcout << " - FE output directory: "<< macrostatelocout << std::endl;
		hcout << " - FE restart directory: "<< macrostatelocres << std::endl;
		hcout << " - FE log directory: "<< macrologloc << std::endl;
	}




	template <int dim>
	void CONTProblem<dim>::set_repositories ()
	{
		if(!file_exists(macrostatelocin)){
			std::cerr << "Missing macroscale or nanoscale input directories." << std::endl;
			exit(1);
		}

		mkdir(macrostatelocout.c_str(), ACCESSPERMS);
		mkdir(macrostatelocres.c_str(), ACCESSPERMS);
		mkdir(macrologloc.c_str(), ACCESSPERMS);

	}



	template <int dim>
	void CONTProblem<dim>::share_scale_bridging_data (ScaleBridgingData &scale_bridging_data)
	{
		int n_updates = scale_bridging_data.update_list.size();
		MPI_Bcast(&n_updates , 1, MPI_INT, 0, world_communicator);
		if (this_world_process != 0) {
			scale_bridging_data.update_list.resize(n_updates);
		}
		MPI_Bcast(&(scale_bridging_data.update_list[0]), n_updates, MPI_QP, 0, world_communicator);
	}



	template <int dim>
	void CONTProblem<dim>::do_timestep ()
	{
		// Begin walltime measuring point
		auto wcts = std::chrono::system_clock::now(); // current time

		// Updating time variable
		present_time += fe_timestep_length;
		++timestep;
		hcout << "Timestep " << timestep << " at time " << present_time
				<< std::endl;
		if (present_time > end_time)
		{
			fe_timestep_length -= (present_time - end_time);
			present_time = end_time;
		}

		newtonstep = 0;

		// Initialisation of timestep variables
		fe_problem->beginstep(timestep, present_time);

		// Solving iteratively the current timestep
		bool continue_newton = false;

		do
		{
			++newtonstep;

			// Might want to simplify that structure now the scale bridging is done outside
			ScaleBridgingData scale_bridging_data;
			fe_problem->solve(newtonstep, scale_bridging_data);
			share_scale_bridging_data(scale_bridging_data);
			
			// O_I
			// Send strains to MD
			//fe_problem->send_strain(scale_bridging_data);

			// S
			// Retrieve stresses
			//fe_problem->receive_stress(scale_bridging_data);

			share_scale_bridging_data(scale_bridging_data);
			continue_newton = fe_problem->check(scale_bridging_data);

		} while (continue_newton);

		fe_problem->endstep();
		
		MPI_Barrier(world_communicator);

		// End time measuring point
		std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
		hcout << "Time for timestep: " << timestep << " is " << wctduration.count() << " seconds\n\n";
	}



	template <int dim>
	void CONTProblem<dim>::run (std::string inputfile)
	{
		// Reading JSON input file (common ones only)
		read_inputs(inputfile);

		hcout << "Building the FE problem:       " << std::endl;

		// Setting repositories for input and creating repositories for outputs
		set_repositories();

		//
		create_qp_mpi_datatype(); // Creates and commits MPI_QP for communicating quadrature point info
																	// between FE and MD solvers

		// Instantiation of the FE problem
		fe_problem = new FEProblem<dim> (world_communicator, world_pcolor, fe_degree, quadrature_formula, n_world_processes);

		// Initialization of time variables
		timestep = start_timestep - 1;
		present_time = timestep*fe_timestep_length;
		end_time = end_timestep*fe_timestep_length;

		// Initialization of MMD must be done before initialization of FE, because FE needs initial
		// materials properties obtained from MMD initialization

		hcout << " Initiation of the Finite Element problem...       " << std::endl;
		fe_problem->init(start_timestep, fe_timestep_length,
										macrostatelocin, macrostatelocout,
										macrostatelocres, macrologloc,
										freq_checkpoint, freq_output_visu, freq_output_lhist, freq_output_lbcforce,
										mdtype,
										twod_mesh_file, extrude_length, extrude_points,
										input_config);

		// Running the solution algorithm of the FE problem
		hcout << "Beginning of incremental solution algorithm:       " << std::endl;
		while (present_time < end_time){
			do_timestep();
		}
		
		delete fe_problem;
	}
}



int main (int argc, char **argv)
{
	try
	{
		using namespace CONT;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		if(argc!=2){
			std::cerr << "Wrong number of arguments, expected: './dealii inputs_dealii.json', but argc is " << argc << std::endl;
			exit(1);
		}

		std::string inputfile = argv[1];
		if(!file_exists(inputfile)){
			std::cerr << "Missing dealii input file." << std::endl;
			exit(1);
		}

		// Begin Global wall-timer
		auto wcts = std::chrono::system_clock::now(); // current wall time

		CONTProblem<3> cont_problem;
		cont_problem.run(inputfile);

		// End of global wall-timer
		std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
		int this_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
		if (this_rank==0) {
		    std::cout << "Overall wall time is " << wctduration.count() << " seconds\n\n";
		}
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
