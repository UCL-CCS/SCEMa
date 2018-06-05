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
	lammps_homogenization (void *lmp, char *location, SymmetricTensor<2,dim>& stresses, SymmetricTensor<4,dim>& stiffnesses, double dts, bool init)
	{
		SymmetricTensor<2,2*dim> tmp;

		char cfile[1024];
		char cline[1024];

		sprintf(cline, "variable locbe string %s/%s", location, "ELASTIC");
		lammps_command(lmp,cline);

		// number of timesteps for averaging
		int nssample = 50;
		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", nssample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", nssample); lammps_command(lmp,cline);

		// number of timesteps for straining
		double strain_rate = 1.0e-4; // in fs^(-1)
		double strain_nrm = 0.20;
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


	// The straining function is ran on every quadrature point which
	// requires a stress_update. Since a quandrature point is only reached*
	// by a subset of processes N, we should automatically see lammps be
	// parallelized on the N processes.
	template <int dim>
	void
	lammps_straining (const SymmetricTensor<2,dim>& strain,
			SymmetricTensor<2,dim>& init_stress,
			SymmetricTensor<2,dim>& stress,
			SymmetricTensor<4,dim>& stiffness,
			std::vector<double>& length,
			char* cellid,
			char* timeid,
			MPI_Comm comm_lammps,
			const char* stateloc,
			const char* logloc,
			std::string mdt,
		    unsigned int repl)
	{
		int me;
		MPI_Comm_rank(comm_lammps, &me);

		// Is this initialization?
		bool init = false;

		// Declaration of run parameters
		// timestep length in fs
		double dts = 2.0;

		// number of timesteps
		double strain_rate = 1.0e-4; // in fs^(-1)
		double strain_nrm = strain.norm();
		int nts = std::ceil(strain_nrm/(dts*strain_rate)/10)*10;

		// number of timesteps between dumps
		int ntsdump = std::ceil(nts/10);

		// Temperature
		double tempt = 300.0;

		// Locations for finding reference LAMMPS files, to store nanostate binary data, and
		// to place LAMMPS log/dump/temporary restart outputs
		char location[1024] = "../box";

		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "%s_%d", mdt.c_str(), repl);
		char initdata[1024];
		sprintf(initdata, "init.%s.bin", mdstate);

		char atomstate[1024];
		sprintf(atomstate, "%s_%d.lammpstrj", mdt.c_str(), repl);

		char replogloc[1024];
		sprintf(replogloc, "%s/R%d", logloc, repl);
		//mkdir(replogloc, ACCESSPERMS);

		char qpreplogloc[1024];
		sprintf(qpreplogloc, "%s/%s.%s", replogloc, timeid, cellid);
		//mkdir(qpreplogloc, ACCESSPERMS);

		char straindata_last[1024];
		sprintf(straindata_last, "last.%s.%s.dump", cellid, mdstate);
		// sprintf(straindata_last, "last.%s.%s.bin", cellid, mdstate);

		char atomdata_last[1024];
		sprintf(atomdata_last, "last.%s.%s", cellid, atomstate);

		char cline[1024];
		char cfile[1024];
		char mfile[1024];

		// Specifying the command line options for screen and log output file
		int nargs = 5;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-screen";
		lmparg[2] = (char *) "none";
		lmparg[3] = (char *) "-log";
		lmparg[4] = new char[1024];
		sprintf(lmparg[4], "%s/log.%s_stress_strain", qpreplogloc, mdt.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,comm_lammps);

		// Passing initial dimension of the box to adjust applied strain
		sprintf(cline, "variable lxbox0 equal %f", length[0]); lammps_command(lmp,cline);
		sprintf(cline, "variable lybox0 equal %f", length[1]); lammps_command(lmp,cline);
		sprintf(cline, "variable lzbox0 equal %f", length[2]); lammps_command(lmp,cline);

		// Passing location for output as variable
		sprintf(cline, "variable mdt string %s", mdt.c_str()); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", qpreplogloc); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", tempt); lammps_command(lmp,cline);

		// Setting dumping of atom positions for post analysis of the MD simulation
		// DO NOT USE CUSTOM DUMP: WRONG ATOM POSITIONS...
		//sprintf(cline, "dump atom_dump all custom %d %s/%s id type xs ys zs vx vy vz ix iy iz", ntsdump, statelocout, atomdata_last); lammps_command(lmp,cline);
		sprintf(cline, "dump atom_dump all atom %d %s/out/%s", ntsdump, stateloc, atomdata_last); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps");
		lammps_file(lmp,cfile);

		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Compute current state data...       " << std::endl;*/

		// Check if a previous state has already been computed specifically for
		// this quadrature point, otherwise use the initial state (which is the
		// last state of this quadrature point)
		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... from previous state data...   " << std::flush;*/
		sprintf(mfile, "%s/out/%s", stateloc, initdata);
		sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

		// Check the presence of a dump file to restart from
		sprintf(mfile, "%s/out/%s", stateloc, straindata_last);
		std::ifstream ifile(mfile);
		if (ifile.good()){
			/*if (me == 0) std::cout << "  specifically computed." << std::endl;*/
			ifile.close();

			sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", mfile); lammps_command(lmp,cline);

			// sprintf(mfile, "%s/%s", statelocout, straindata_last);
			// sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

			sprintf(cline, "print 'specifically computed'"); lammps_command(lmp,cline);
		}
		else{
			/*if (me == 0) std::cout << "  initially computed." << std::endl;*/

			// sprintf(mfile, "%s/%s", statelocout, initdata);
			// sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

			sprintf(cline, "print 'initially computed'"); lammps_command(lmp,cline);
		}

		sprintf(cline, "variable dts equal %f", dts); lammps_command(lmp,cline);
		sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				sprintf(cline, "variable eeps_%d%d equal %.6e", k, l, strain[k][l]/(nts*dts));
				lammps_command(lmp,cline);
			}

		// Run the NEMD simulations of the strained box
		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... reading and executing in.strain.lammps.       " << std::endl;*/
		sprintf(cfile, "%s/%s", location, "in.strain.lammps");
		lammps_file(lmp,cfile);

		// Unetting dumping of atom positions
		sprintf(cline, "undump atom_dump"); lammps_command(lmp,cline);

		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;*/
		// Save data to specific file for this quadrature point
		sprintf(cline, "write_dump all custom %s/out/%s id type xs ys zs vx vy vz ix iy iz", stateloc, straindata_last); lammps_command(lmp,cline);

		// close down LAMMPS
		delete lmp;

		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;*/

		// Creating LAMMPS instance
		sprintf(lmparg[4], "%s/log.%s_homogenization", qpreplogloc, mdt.c_str());
		lmp = new LAMMPS(nargs,lmparg,comm_lammps);

        sprintf(cline, "variable loco string %s", qpreplogloc); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", tempt); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps");
		lammps_file(lmp,cfile);

		sprintf(mfile, "%s/out/%s", stateloc, initdata);
		sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

		sprintf(mfile, "%s/out/%s", stateloc, straindata_last);
		sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", mfile); lammps_command(lmp,cline);
		//sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

		sprintf(cline, "variable dts equal %f", dts); lammps_command(lmp,cline);

		// Compute the secant stiffness tensor at the given stress/strain state
		lammps_homogenization<dim>(lmp, location, stress, stiffness, dts, init);

		// Cleaning initial offset of stresses
		stress -= init_stress;

		// close down LAMMPS
		delete lmp;
	}




	template <int dim>
	class MDProblem
	{
	public:
		MDProblem (const char* mslocout, const char* nsloc, const char* nsloclog);
		~MDProblem ();
		void run(char* ctime, char* ccell, const char* cmat, unsigned int repl);

	private:

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		const char*                         macrostatelocout;
		const char*                         nanostateloc;
		const char*                         nanologloc;

		ConditionalOStream 					hcout;

	};



	template <int dim>
	MDProblem<dim>::MDProblem (const char* mslocout, const char* nsloc, const char* nsloclog)
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		macrostatelocout (mslocout),
		nanostateloc (nsloc),
		nanologloc (nsloclog),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	MDProblem<dim>::~MDProblem ()
	{}


	template <int dim>
	void MDProblem<dim>::run (char* ctime, char* ccell, const char* cmat, unsigned int repl)
	{
		if(this_world_process == 0) std::cout << "Number of processes assigned: " << n_world_processes << std::endl;  

		SymmetricTensor<2,dim> loc_strain;
		SymmetricTensor<2,dim> loc_rep_stress;

		char filename[1024];

		SymmetricTensor<4,dim> loc_rep_stiffness;
		SymmetricTensor<2,dim> init_rep_stress;
		std::vector<double> init_rep_length (dim);

		// Arguments of the secant stiffness computation
		sprintf(filename, "%s/init.%s_%d.stress", macrostatelocout, cmat, repl);
		read_tensor<dim>(filename, init_rep_stress);

		// Providing initial box dimension to adjust the strain tensor
		sprintf(filename, "%s/init.%s_%d.length", macrostatelocout, cmat, repl);
		read_tensor<dim>(filename, init_rep_length);

		// Argument of the MD simulation: strain to apply
		sprintf(filename, "%s/last.%s.upstrain", macrostatelocout, ccell);
		read_tensor<dim>(filename, loc_strain);

		// Then the lammps function instanciates lammps, starting from an initial
		// microstructure and applying the complete new_strain or starting from
		// the microstructure at the old_strain and applying the difference between
		// the new_ and _old_strains, returns the new_stress state.
		lammps_straining<dim> (loc_strain,
				init_rep_stress,
				loc_rep_stress,
				loc_rep_stiffness,
				init_rep_length,
				ccell,
				ctime,
				world_communicator,
				nanostateloc,
				nanologloc,
				cmat,
				repl);

		if(this_world_process == 0)
		{
			std::cout << " \t" << ccell <<"-"<< repl << " \t" << std::flush;

			/*sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout, ccell, repl);
			write_tensor<dim>(filename, loc_rep_stiffness);*/

			sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout, ccell, repl);
			write_tensor<dim>(filename, loc_rep_stress);
		}
	}
}






int main (int argc, char **argv)
{
	try
	{
		using namespace MD;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		char* ctime= argv[1];
		char* ccell= argv[2];
		std::string cmat = argv[3];
		unsigned int repl = atoi(argv[4]);
		std::string mslocout = argv[5];
		std::string nsloc = argv[6];
		std::string nsloclog = argv[7];
		//std::cout << ctime << " " << ccell << " " << cmat << " " << repl << " " << mslocout << " " << nslocout << " " << nsloclog << std::endl;
		MDProblem<3> md_problem (mslocout.c_str(),nsloc.c_str(),nsloclog.c_str());

		md_problem.run(ctime, ccell, cmat.c_str(), repl);
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
