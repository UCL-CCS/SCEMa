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
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "library.h"
#include "atom.h"

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "headers/read_write.h"
#include "headers/anmd_problem.h"

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/mpi.h>

int main (int argc, char **argv)
{
	try
	{
		using namespace HMM;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		if(argc!=13){
			std::cerr << "Wrong number of arguments, expected: "
					  << "'./analyse_md cellid timeid cellmat statelocout statelocres"
					  << " qpreplogloc scriptsloc repl"
					  << "md_timestep_length md_temperature md_nsteps_sample md_force_field"
					  << ", but argc is " << argc << std::endl;
			exit(1);
		}

		int n_world_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
		int this_world_process = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
		if(this_world_process == 0) std::cout << "Number of processes assigned: "
										      << n_world_processes << std::endl;

		std::string cellid = argv[1];
		std::string timeid = argv[2];
		std::string cellmat = argv[3];

		std::string statelocout = argv[4];
		std::string statelocres = argv[5];
		std::string qpreplogloc = argv[6];
		std::string scriptsloc = argv[7];

		unsigned int repl = std::stoi(argv[8]);

		double md_timestep_length = std::stod(argv[9]);
		double md_temperature = std::stod(argv[10]);
		unsigned int md_nsteps_sample = std::stoi(argv[11]);
		std::string md_force_field = argv[12];

		if(this_world_process == 0) std::cout << "List of arguments: "
											  << cellid << " " << timeid << " " << cellmat
											  << " " << statelocout << " " << statelocres <<  " " << qpreplogloc
											  << " " << scriptsloc
											  << " " << repl << " " << md_timestep_length << " " << md_temperature
											  << " " << md_nsteps_sample <<  " " << md_force_field
											  << std::endl;

		ANMDProblem<3> anmd_problem (MPI_COMM_WORLD, 0);

		anmd_problem.analyse(cellid, timeid, cellmat, statelocout, statelocres,
					   qpreplogloc, scriptsloc, repl, md_timestep_length,
					   md_temperature, md_nsteps_sample, md_force_field);
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
