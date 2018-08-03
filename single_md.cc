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
#include "headers/md_problem.h"

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

		if(argc!=18){
			std::cerr << "Wrong number of arguments, expected: "
					  << "'./single_md cellid timeid cellmat statelocout statelocres"
					  << "loglochom qpreplogloc scriptsloc macrostatelocout repl"
					  << "md_timestep_length md_temperature md_nsteps_sample md_strain_rate"
					  << "output_homog checkpoint_save'"
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
		std::string loglochom = argv[6];
		std::string qpreplogloc = argv[7];
		std::string scriptsloc = argv[8];

		std::string straininputfile = argv[9];
		std::string stressoutputfile = argv[10];

		unsigned int repl = std::stoi(argv[11]);

		double md_timestep_length = std::stod(argv[12]);
		double md_temperature = std::stod(argv[13]);
		unsigned int md_nsteps_sample = std::stoi(argv[14]);
		double md_strain_rate = std::stod(argv[15]);

		bool output_homog = std::stoi(argv[16]);
		bool checkpoint_save = std::stoi(argv[17]);

		if(this_world_process == 0) std::cout << "List of arguments: "
											  << cellid << " " << timeid << " " << cellmat << " " << statelocout
											  << " " << statelocres << " " << loglochom << " " << qpreplogloc
											  << " " << scriptsloc << " " << straininputfile << " " << stressoutputfile
											  << " " << repl << " " << md_timestep_length << " " << md_temperature
											  << " " << md_nsteps_sample << " " << md_strain_rate << " " << output_homog
											  << " " << checkpoint_save
											  << std::endl;

		MDProblem<3> md_problem (MPI_COMM_WORLD, 0);

		md_problem.run(cellid, timeid, cellmat, statelocout, statelocres, loglochom,
					   qpreplogloc, scriptsloc, straininputfile, stressoutputfile, repl, md_timestep_length,
					   md_temperature, md_nsteps_sample, md_strain_rate, output_homog, checkpoint_save);
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
