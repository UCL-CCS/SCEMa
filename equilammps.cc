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

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "headers/read_write.h"
#include "headers/tensor_calc.h"
#include "headers/eqmd_sync.h"

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/mpi.h>

namespace HMM
{
	using namespace dealii;

	template <int dim>
	class EMDProblem
	{
	public:
		EMDProblem ();
		~EMDProblem ();
		void run (std::string inputfile);

	private:
		void read_inputs(std::string inputfile);

		void set_global_communicators ();
		void set_repositories ();

		EQMDSync<dim> 						*mmd_problem = NULL;

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		MPI_Comm 							mmd_communicator;
		int 								n_mmd_processes;
		int									root_mmd_process;
		int 								this_mmd_process;
		int 								mmd_pcolor;

		unsigned int						machine_ppn;
		unsigned int						batch_nnodes_min;

		ConditionalOStream 					hcout;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;
		Tensor<1,dim> 						cg_dir;

		bool								use_pjm_scheduler;

		double								md_timestep_length;
		double								md_temperature;
		int									md_nsteps_sample;
		int									md_nsteps_equil;
		double								md_strain_rate;
		double								md_strain_ampl;
		std::string							md_force_field;

		std::string                         nanostatelocin;
		std::string							nanostatelocout;
		std::string							nanologloc;
		std::string							nanologloctmp;

		std::string							md_scripts_directory;


	};



	template <int dim>
	EMDProblem<dim>::EMDProblem ()
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	EMDProblem<dim>::~EMDProblem ()
	{}




	template <int dim>
	void EMDProblem<dim>::read_inputs (std::string inputfile)
	{
	    using boost::property_tree::ptree;

	    std::ifstream jsonFile(inputfile);
	    ptree pt;
	    try{
		    read_json(jsonFile, pt);
	    }
	    catch (const boost::property_tree::json_parser::json_parser_error& e)
	    {
	        hcout << "Invalid JSON HMM input file (" << inputfile << ")" << std::endl;  // Never gets here
	    }

	    // Scale-bridging parameters
	    use_pjm_scheduler = std::stoi(bptree_read(pt, "scale-bridging", "use pjm scheduler"));

		// Atomic input, output, restart and log location
		nanostatelocin = bptree_read(pt, "directory structure", "nanoscale input");
		nanologloc = bptree_read(pt, "directory structure", "nanoscale log");

		// Molecular dynamics material data
		nrepl = std::stoi(bptree_read(pt, "molecular dynamics material", "number of replicas"));
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
				get_subbptree(pt, "molecular dynamics material").get_child("list of materials.")) {
			mdtype.push_back(v.second.data());
		}
		// Direction to which all MD data are rotated to, to later ease rotation in the FE problem. The
		// replicas results are rotated to this referential before ensemble averaging, and the continuum
		// tensors are rotated to this referential from the microstructure given orientation
		std::vector<double> tmp_dir;
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
				get_subbptree(pt, "molecular dynamics material").get_child("rotation common ground vector.")) {
			tmp_dir.push_back(std::stod(v.second.data()));
		}
		if(tmp_dir.size()==dim){
			for(unsigned int imd=0; imd<dim; imd++){
				cg_dir[imd] = tmp_dir[imd];
			}
		}

		// Molecular dynamics simulation parameters
		md_timestep_length = std::stod(bptree_read(pt, "molecular dynamics parameters", "timestep length"));
		md_temperature = std::stod(bptree_read(pt, "molecular dynamics parameters", "temperature"));
		md_nsteps_sample = std::stoi(bptree_read(pt, "molecular dynamics parameters", "number of sampling steps"));
		md_nsteps_equil = std::stoi(bptree_read(pt, "molecular dynamics parameters", "number of equilibration steps"));
		md_strain_rate = std::stod(bptree_read(pt, "molecular dynamics parameters", "strain rate"));
		md_strain_ampl = std::stod(bptree_read(pt, "molecular dynamics parameters", "strain amplitude"));
		md_force_field = bptree_read(pt, "molecular dynamics parameters", "force field");
		md_scripts_directory = bptree_read(pt, "molecular dynamics parameters", "scripts directory");

		// Computational resources
		machine_ppn = std::stoi(bptree_read(pt, "computational resources", "machine cores per node"));
		batch_nnodes_min = std::stoi(bptree_read(pt, "computational resources", "minimum nodes per MD simulation"));

		// Print a recap of all the parameters...
		hcout << "Parameters listing:" << std::endl;
		hcout << " - Use Pilot Job Manager to schedule MD jobs: "<< use_pjm_scheduler << std::endl;
		hcout << " - Number of replicas: "<< nrepl << std::endl;
		hcout << " - List of material names: "<< std::flush;
		for(unsigned int imd=0; imd<mdtype.size(); imd++) hcout << " " << mdtype[imd] << std::flush; hcout << std::endl;;
		hcout << " - Direction use as a common ground/referential to transfer data between nano- and micro-structures : "<< std::flush;
		for(unsigned int imd=0; imd<dim; imd++) hcout << " " << cg_dir[imd] << std::flush; hcout << std::endl;;
		hcout << " - MD timestep duration: "<< md_timestep_length << std::endl;
		hcout << " - MD thermostat temperature: "<< md_temperature << std::endl;
		hcout << " - MD deformation rate: "<< md_strain_rate << std::endl;
		hcout << " - MD deformation amplitude for homogenization of stiffness: "<< md_strain_ampl << std::endl;
		hcout << " - MD number of sampling steps: "<< md_nsteps_sample << std::endl;
		hcout << " - MD number of equilibration steps: "<< md_nsteps_equil << std::endl;
		hcout << " - MD force field type: "<< md_force_field << std::endl;
		hcout << " - MD scripts directory (contains in.set, in.strain, ELASTIC/, ffield parameters): "<< md_scripts_directory << std::endl;
		hcout << " - Number of cores per node on the machine: "<< machine_ppn << std::endl;
		hcout << " - Minimum number of nodes per MD simulation: "<< batch_nnodes_min << std::endl;
		hcout << " - MD input directory: "<< nanostatelocin << std::endl;
		hcout << " - MD log directory: "<< nanologloc << std::endl;
	}




	template <int dim>
	void EMDProblem<dim>::set_global_communicators ()
	{

		//Setting up LAMMPS communicator and related variables
		root_mmd_process = 0;
		n_mmd_processes = n_world_processes;
		// Color set above 0 for processors that are going to be used
		mmd_pcolor = MPI_UNDEFINED;
		if (this_world_process >= root_mmd_process &&
				this_world_process < root_mmd_process + n_mmd_processes) mmd_pcolor = 0;
		else mmd_pcolor = 1;

		MPI_Comm_split(world_communicator, mmd_pcolor, this_world_process, &mmd_communicator);
		MPI_Comm_rank(mmd_communicator, &this_mmd_process);
	}




	template <int dim>
	void EMDProblem<dim>::set_repositories ()
	{
		if(!file_exists(nanostatelocin)){
			std::cerr << "Missing macroscale or nanoscale input directories." << std::endl;
			exit(1);
		}

		mkdir(nanologloc.c_str(), ACCESSPERMS);
		nanologloctmp = nanologloc+"/tmp"; mkdir(nanologloctmp.c_str(), ACCESSPERMS);

		char fnset[1024]; sprintf(fnset, "%s/in.set.lammps", md_scripts_directory.c_str());
		char fnstrain[1024]; sprintf(fnstrain, "%s/in.strain.lammps", md_scripts_directory.c_str());
		char fnelastic[1024]; sprintf(fnelastic, "%s/ELASTIC", md_scripts_directory.c_str());

		if(!file_exists(fnset) || !file_exists(fnstrain) || !file_exists(fnelastic)){
			std::cerr << "Missing some MD input scripts for executing LAMMPS simulation (in 'box' directory)." << std::endl;
			exit(1);
		}
	}



	template <int dim>
	void EMDProblem<dim>::run (std::string inputfile)
	{
		// Reading JSON input file (common ones only)
		read_inputs(inputfile);

		hcout << "Building the HMM problem:       " << std::endl;

		// Set the dealii communicator using a limited amount of available processors
		// because dealii fails if processors do not have assigned cells. Plus, dealii
		// might not scale indefinitely
		set_global_communicators();

		// Setting repositories for input and creating repositories for outputs
		set_repositories();

		// Instantiation of the MMD Problem
		if(mmd_pcolor==0) mmd_problem = new EQMDSync<dim> (mmd_communicator, mmd_pcolor);

		MPI_Barrier(world_communicator);

		hcout << " Equilibration of Multiple Molecular Dynamics systems...       " << std::endl;
		if(mmd_pcolor==0) mmd_problem->equilibrate(md_timestep_length, md_temperature,
											   md_nsteps_sample, md_nsteps_equil, md_strain_rate, md_strain_ampl,
											   md_force_field, nanostatelocin, nanologloc,
											   nanologloctmp,
											   md_scripts_directory,
											   batch_nnodes_min, machine_ppn, mdtype, cg_dir, nrepl,
											   use_pjm_scheduler);

		MPI_Barrier(world_communicator);

		if(mmd_pcolor==0) delete mmd_problem;
	}
}



int main (int argc, char **argv)
{
	try
	{
		using namespace HMM;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		if(argc!=2){
			std::cerr << "Wrong number of arguments, expected: './dealammps inputs_equilammps.json', but argc is " << argc << std::endl;
			exit(1);
		}

		std::string inputfile = argv[1];
		if(!file_exists(inputfile)){
			std::cerr << "Missing HMM input file." << std::endl;
			exit(1);
		}

		EMDProblem<3> emd_problem;
		emd_problem.run(inputfile);
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
