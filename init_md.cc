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

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/mpi.h>

namespace MD
{
	using namespace dealii;
	using namespace LAMMPS_NS;

	bool file_exists(const char* file) {
		struct stat buf;
		return (stat(file, &buf) == 0);
	}


	template <int dim>
	inline
	void
	read_tensor (char *filename, std::vector<double> &tensor)
	{
		std::ifstream ifile;

		ifile.open (filename);
		if (ifile.is_open())
		{
			for(unsigned int k=0;k<tensor.size();k++)
				{
					char line[1024];
					if(ifile.getline(line, sizeof(line)))
						tensor[k] = std::strtod(line, NULL);
				}
			ifile.close();
		}
		else std::cout << "Unable to open" << filename << " to read it" << std::endl;
	}

	template <int dim>
	inline
	void
	read_tensor (char *filename, SymmetricTensor<2,dim> &tensor)
	{
		std::ifstream ifile;

		ifile.open (filename);
		if (ifile.is_open())
		{
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
				{
					char line[1024];
					if(ifile.getline(line, sizeof(line)))
						tensor[k][l] = std::strtod(line, NULL);
				}
			ifile.close();
		}
		else std::cout << "Unable to open" << filename << " to read it" << std::endl;
	}

	template <int dim>
	inline
	void
	read_tensor (char *filename, SymmetricTensor<4,dim> &tensor)
	{
		std::ifstream ifile;

		ifile.open (filename);
		if (ifile.is_open())
		{
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					for(unsigned int m=0;m<dim;m++)
						for(unsigned int n=m;n<dim;n++)
						{
							char line[1024];
							if(ifile.getline(line, sizeof(line)))
								tensor[k][l][m][n]= std::strtod(line, NULL);
						}
			ifile.close();
		}
		else std::cout << "Unable to open" << filename << " to read it..." << std::endl;
	}

	template <int dim>
	inline
	void
	write_tensor (char *filename, std::vector<double> &tensor)
	{
		std::ofstream ofile;

		ofile.open (filename);
		if (ofile.is_open())
		{
			for(unsigned int k=0;k<tensor.size();k++)
					//std::cout << std::setprecision(16) << tensor[k][l] << std::endl;
					ofile << std::setprecision(16) << tensor[k] << std::endl;
			ofile.close();
		}
		else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
	}

	template <int dim>
	inline
	void
	write_tensor (char *filename, SymmetricTensor<2,dim> &tensor)
	{
		std::ofstream ofile;

		ofile.open (filename);
		if (ofile.is_open())
		{
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					//std::cout << std::setprecision(16) << tensor[k][l] << std::endl;
					ofile << std::setprecision(16) << tensor[k][l] << std::endl;
			ofile.close();
		}
		else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
	}

	template <int dim>
	inline
	void
	write_tensor (char *filename, SymmetricTensor<4,dim> &tensor)
	{
		std::ofstream ofile;

		ofile.open (filename);
		if (ofile.is_open())
		{
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					for(unsigned int m=0;m<dim;m++)
						for(unsigned int n=m;n<dim;n++)
							ofile << std::setprecision(16) << tensor[k][l][m][n] << std::endl;
			ofile.close();
		}
		else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
	}

	// Computes the stress tensor and the complete tanget elastic stiffness tensor
	template <int dim>
	void
	lammps_homogenization (void *lmp, char *location, SymmetricTensor<2,dim>& stresses, SymmetricTensor<4,dim>& stiffnesses, bool init)
	{
		SymmetricTensor<2,2*dim> tmp;

		char cfile[1024];
		char cline[1024];

		sprintf(cline, "variable locbe string %s/%s", location, "ELASTIC");
		lammps_command(lmp,cline);

		// Timestep length in fs
		double dts = 2.0;

		// number of timesteps for averaging
		int nssample = 200;
		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", nssample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", nssample); lammps_command(lmp,cline);

		// number of timesteps for straining
		double strain_rate = 1.0e-5; // in fs^(-1)
		double strain_nrm = 0.005;
		int nsstrain = std::ceil(strain_nrm/(dts*strain_rate)/10)*10;
		// For v_sound_PE = 2000 m/s, l_box=8nm, strain_perturbation=0.005, and dts=2.0fs
		// the min number of straining steps is 10
		sprintf(cline, "variable nsstrain  equal %d", nsstrain); lammps_command(lmp,cline);

		// Set strain perturbation amplitude
		sprintf(cline, "variable up equal %f", strain_nrm); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress tensor
		sprintf(cfile, "%s/%s", location, "ELASTIC/in.elastic.lammps");
		lammps_file(lmp,cfile);

		// Filling 3x3 stress tensor and conversion from ATM to Pa
		// Useless at the moment, since it cannot be used in the Newton-Raphson algorithm.
		// The MD evaluated stress is flucutating too much (few MPa), therefore prevents
		// the iterative algorithm to converge...
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				char vcoef[1024];
				sprintf(vcoef, "pp%d%d", k+1, l+1);
				stresses[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*(-1.0)*1.01325e+05;
			}

		if(init){
			// Using a routine based on the example ELASTIC/ to compute the stiffness tensor
			sprintf(cfile, "%s/%s", location, "ELASTIC/in.homog.lammps");
			lammps_file(lmp,cfile);

			// Filling the 6x6 Voigt Sitffness tensor with its computed as variables
			// by LAMMPS and conversion from GPa to Pa
			for(unsigned int k=0;k<2*dim;k++)
				for(unsigned int l=k;l<2*dim;l++)
				{
					char vcoef[1024];
					sprintf(vcoef, "C%d%dall", k+1, l+1);
					tmp[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;
				}

			// Write test... (on the data returned by lammps)

			// Conversion of the 6x6 Voigt Stiffness Tensor into the 3x3x3x3
			// Standard Stiffness Tensor
			for(unsigned int i=0;i<2*dim;i++)
			{
				int k, l;
				if     (i==(3+0)){k=0; l=1;}
				else if(i==(3+1)){k=0; l=2;}
				else if(i==(3+2)){k=1; l=2;}
				else  /*(i<3)*/  {k=i; l=i;}


				for(unsigned int j=0;j<2*dim;j++)
				{
					int m, n;
					if     (j==(3+0)){m=0; n=1;}
					else if(j==(3+1)){m=0; n=2;}
					else if(j==(3+2)){m=1; n=2;}
					else  /*(j<3)*/  {m=j; n=j;}

					stiffnesses[k][l][m][n]=tmp[i][j];
				}
			}
		}

	}


	// The initiation, namely the preparation of the data from which will
	// be ran the later tests at every quadrature point, should be ran on
	// as many processes as available, since it will be the only on going
	// task at the time it will be called.
	template <int dim>
	void
	lammps_initiation (SymmetricTensor<2,dim>& stress,
			SymmetricTensor<4,dim>& stiffness,
			std::vector<double>& lbox0,
			MPI_Comm comm_lammps,
			const char* statelocin,
			const char* statelocout,
			const char* logloc,
			std::string mdt,
			unsigned int repl)
	{
		// Is this initialization?
		bool init = true;

		// Timestep length in fs
		double dts = 1.0;
		// Number of timesteps factor
		int nsinit = 20000;
		// Temperature
		double tempt = 300.0;

		// Locations for finding reference LAMMPS files, to store nanostate binary data, and
		// to place LAMMPS log/dump/temporary restart outputs
		char location[1024] = "../box";

		char locdata[1024];
		sprintf(locdata, "%s/data/%s_%d.data", statelocin, mdt.c_str(), repl);

		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "%s_%d.bin", mdt.c_str(), repl);
		char initdata[1024];
		sprintf(initdata, "init.%s", mdstate);

		int me;
		MPI_Comm_rank(comm_lammps, &me);

		// Repositories creation and checking...
		char replogloc[1024];
		sprintf(replogloc, "%s/R%d", logloc, repl);
		mkdir(replogloc, ACCESSPERMS);

		char inireplogloc[1024];
		sprintf(inireplogloc, "%s/%s", replogloc, "init");
		mkdir(inireplogloc, ACCESSPERMS);

		char cfile[1024];
		char cline[1024];
		char sfile[1024];

		// Specifying the command line options for screen and log output file
		int nargs = 5;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-screen";
		lmparg[2] = (char *) "none";
		lmparg[3] = (char *) "-log";
		lmparg[4] = new char[1024];
		sprintf(lmparg[4], "%s/log.%s_heatup_cooldown", inireplogloc, mdt.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,comm_lammps);

		// Setting run parameters
		sprintf(cline, "variable dts equal %f", dts); lammps_command(lmp,cline);
		sprintf(cline, "variable nsinit equal %d", nsinit); lammps_command(lmp,cline);

		// Passing location for input and output as variables
		sprintf(cline, "variable mdt string %s", mdt.c_str()); lammps_command(lmp,cline);
		sprintf(cline, "variable locd string %s", locdata); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", inireplogloc); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps"); lammps_file(lmp,cfile);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", tempt); lammps_command(lmp,cline);
		sprintf(cline, "variable sseed equal 1234"); lammps_command(lmp,cline);

		// Check if 'init.PE.bin' has been computed already
		sprintf(sfile, "%s/%s", statelocin, initdata);
		bool state_exists = file_exists(sfile);

		if (!state_exists)
		{
			if (me == 0) std::cout << "(MD - init - type " << mdt << " - repl " << repl << ") "
					<< "Compute state data...       " << std::endl;
			// Compute initialization of the sample which minimizes the free energy,
			// heat up and finally cool down the sample.
			sprintf(cfile, "%s/%s", location, "in.init.lammps"); lammps_file(lmp,cfile);
		}
		else
		{
			if (me == 0) std::cout << "(MD - init - type " << mdt << " - repl " << repl << ") "
					<< "Reuse of state data...       " << std::endl;
			// Reload from previously computed initial preparation (minimization and
			// heatup/cooldown), this option shouldn't remain, as in the first step the
			// preparation should always be computed.
			sprintf(cline, "read_restart %s/%s", statelocin, initdata); lammps_command(lmp,cline);
		}

		// Storing initial dimensions after initiation
		char lname[1024];
		sprintf(lname, "lxbox0");
		sprintf(cline, "variable tmp equal 'lx'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		lbox0[0] = *((double *) lammps_extract_variable(lmp,lname,NULL));
		sprintf(lname, "lybox0");
		sprintf(cline, "variable tmp equal 'ly'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		lbox0[1] = *((double *) lammps_extract_variable(lmp,lname,NULL));
		sprintf(lname, "lzbox0");
		sprintf(cline, "variable tmp equal 'lz'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		lbox0[2] = *((double *) lammps_extract_variable(lmp,lname,NULL));

		// Saving nanostate at the end of initiation
		if (me == 0) std::cout << "(MD - init - type " << mdt << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;
		sprintf(cline, "write_restart %s/%s", statelocout, initdata); lammps_command(lmp,cline);

		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				sprintf(cline, "variable eeps_%d%d equal %.6e", k, l, 0.0);
				lammps_command(lmp,cline);
			}

		if (me == 0) std::cout << "(MD - init - type " << mdt << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;
		// Compute secant stiffness operator and initial stresses
		lammps_homogenization<dim>(lmp, location, stress, stiffness, init);

		// close down LAMMPS
		delete lmp;

	}



	template <int dim>
	class INIProblem
	{
	public:
		INIProblem (const char* mslocout, const char* nslocin, const char* nslocout, const char* nsloclog);
		~INIProblem ();
		void run(const char* cmat, unsigned int repl);

	private:

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		const char*                         macrostatelocout;
		const char*                         nanostatelocin;
		const char*                         nanostatelocout;
		const char*                         nanologloc;

		ConditionalOStream 					hcout;

	};



	template <int dim>
	INIProblem<dim>::INIProblem (const char* mslocout, const char* nslocin, const char* nslocout, const char* nsloclog)
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		macrostatelocout (mslocout),
		nanostatelocin (nslocin),
		nanostatelocout (nslocout),
		nanologloc (nsloclog),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	INIProblem<dim>::~INIProblem ()
	{}


	template <int dim>
	void INIProblem<dim>::run (const char* cmat, unsigned int repl)
	{

		std::vector<double> 				initial_length (dim);
		SymmetricTensor<2,dim> 				initial_stress_tensor;
		SymmetricTensor<4,dim> 				initial_stiffness_tensor;

		lammps_initiation<dim> (initial_stress_tensor, initial_stiffness_tensor, initial_length, world_communicator,
										nanostatelocin, nanostatelocout, nanologloc, cmat, repl);

		if(this_world_process == 0)
		{

			char macrofilenameout[1024];
			sprintf(macrofilenameout, "%s/init.%s_%d.stiff", macrostatelocout, cmat, repl);
			char macrofilenameoutstress[1024];
			sprintf(macrofilenameoutstress, "%s/init.%s_%d.stress", macrostatelocout, cmat, repl);
			char macrofilenameoutlength[1024];
			sprintf(macrofilenameoutlength, "%s/init.%s_%d.length", macrostatelocout, cmat, repl);

			write_tensor<dim>(macrofilenameout, initial_stiffness_tensor);
			write_tensor<dim>(macrofilenameoutstress, initial_stress_tensor);
			write_tensor<dim>(macrofilenameoutlength, initial_length);
		}
	}
}




int main (int argc, char **argv)
{
	try
	{
		using namespace MD;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		std::string cmat = argv[1];
		unsigned int repl = int(*argv[2]);
		std::string mslocout = argv[3];
		std::string nslocin = argv[4];
		std::string nslocout = argv[5];
		std::string nsloclog = argv[6];

		INIProblem<3> ini_problem (mslocout.c_str(),nslocin.c_str(),nslocout.c_str(),nsloclog.c_str());

		ini_problem.run(cmat.c_str(), repl);
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
