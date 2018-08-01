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




	template <int dim>
	class MDProblem
	{
	public:
		MDProblem ();
		~MDProblem ();
		void run(std::string cid, std::string 	tid, std::string cmat,
				  std::string slocout, std::string slocres, std::string llochom,
				  std::string qplogloc, std::string scrloc, std::string mslocout,
				  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
				  double mdss, bool outhom, bool checksav);

	private:

		void lammps_straining();

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		ConditionalOStream 					hcout;

		SymmetricTensor<2,dim> 				loc_rep_strain;
		SymmetricTensor<2,dim> 				loc_rep_stress;
		SymmetricTensor<2,dim> 				init_rep_stress;
		std::vector<double> 				init_rep_length;

		std::string 						cellid;
		std::string 						timeid;
		std::string 						cellmat;

		std::string 						statelocout;
		std::string 						statelocres;
		std::string 						loglochom;
		std::string 						qpreplogloc;
		std::string 						scriptsloc;

		std::string 						macrostatelocout;

		unsigned int 						repl;

		double								md_timestep_length;
		double								md_temperature;
		unsigned int 						md_nsteps_sample;
		double								md_strain_rate;

		bool								output_homog;
		bool								checkpoint_save;

	};



	template <int dim>
	MDProblem<dim>::MDProblem ()
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	MDProblem<dim>::~MDProblem ()
	{}



	// The straining function is ran on every quadrature point which
	// requires a stress_update. Since a quandrature point is only reached*
	// by a subset of processes N, we should automatically see lammps be
	// parallelized on the N processes.
	template <int dim>
	void MDProblem<dim>::lammps_straining ()
	{
		//char locff[1024]; /*reaxff*/
		//sprintf(locff, "%s/ffield.reax.2", scriptsloc.c_str(); /*reaxff*/

		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "%s_%d", cellmat.c_str(), repl);

		char initdata[1024];
		sprintf(initdata, "%s/init.%s.bin", statelocout.c_str(), mdstate);

		char straindata_last[1024];
		sprintf(straindata_last, "%s/last.%s.%s.dump", statelocout.c_str(), cellid, mdstate);
		// sprintf(straindata_last, "%s/last.%s.%s.bin", statelocout.c_str(), cellid, mdstate);

		char straindata_time[1024];
		sprintf(straindata_time, "%s/%s.%s.%s.dump", statelocres.c_str(), timeid, cellid, mdstate);
		// sprintf(straindata_lcts, "%s/lcts.%s.%s.bin", statelocres.c_str(), cellid, mdstate);

		char straindata_lcts[1024];
		sprintf(straindata_lcts, "%s/lcts.%s.%s.dump", statelocres.c_str(), cellid, mdstate);
		// sprintf(straindata_lcts, "%s/lcts.%s.%s.bin", statelocres.c_str(), cellid, mdstate);

		char homogdata_time[1024];
		sprintf(homogdata_time, "%s/%s.%s.%s.lammpstrj", loglochom.c_str(), timeid, cellid, mdstate);

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
		sprintf(lmparg[4], "%s/log.%s_stress_strain", qpreplogloc.c_str(), cellmat.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,world_communicator);

		// Passing initial dimension of the box to adjust applied strain
		sprintf(cline, "variable lxbox0 equal %f", init_rep_length[0]); lammps_command(lmp,cline);
		sprintf(cline, "variable lybox0 equal %f", init_rep_length[1]); lammps_command(lmp,cline);
		sprintf(cline, "variable lzbox0 equal %f", init_rep_length[2]); lammps_command(lmp,cline);

		// Passing location for output as variable
		sprintf(cline, "variable mdt string %s", cellmat.c_str()); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", qpreplogloc.c_str()); lammps_command(lmp,cline);
		//sprintf(cline, "variable locf string %s", locff); lammps_command(lmp,cline); /*reaxff*/

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_temperature); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.set.lammps");
		lammps_file(lmp,cfile);

		/*hcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Compute current state data...       " << std::endl;*/

		// Check if a previous state has already been computed specifically for
		// this quadrature point, otherwise use the initial state (which is the
		// last state of this quadrature point)
		/*hcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... from previous state data...   " << std::flush;*/
		//sprintf(mfile, "%s", initdata); /*reaxff*/
		//sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline); /*reaxff*/

		// Check the presence of a dump file to restart from
		sprintf(mfile, "%s", straindata_last);
		std::ifstream ifile(mfile);
		if (ifile.good()){
			/*hcout << "  specifically computed." << std::endl;*/
			ifile.close();

			//sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", mfile); lammps_command(lmp,cline); /*reaxff*/

			sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline); /*opls*/

			sprintf(cline, "print 'specifically computed'"); lammps_command(lmp,cline);
		}
		else{
			/*hcout << "  initially computed." << std::endl;*/

			sprintf(mfile, "%s", initdata); /*opls*/
			sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline); /*opls*/

			sprintf(cline, "print 'initially computed'"); lammps_command(lmp,cline);
		}

		// Number of timesteps in the MD simulation, enforcing at least one.
		int nts = std::max(int(std::ceil(loc_rep_strain.norm()/(md_timestep_length*md_strain_rate)/10)*10),1);

		sprintf(cline, "variable dts equal %f", md_timestep_length); lammps_command(lmp,cline);
		sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				sprintf(cline, "variable eeps_%d%d equal %.6e", k, l, loc_rep_strain[k][l]/(nts*md_timestep_length));
				lammps_command(lmp,cline);
			}

		// Run the NEMD simulations of the strained box
		/*hcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... reading and executing in.strain.lammps.       " << std::endl;*/
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.strain.lammps");
		lammps_file(lmp,cfile);

		/*hcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;*/
		// Save data to specific file for this quadrature point
		sprintf(cline, "write_restart %s", straindata_last); lammps_command(lmp,cline); /*opls*/
		//sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_last); lammps_command(lmp,cline); /*reaxff*/

		if(checkpoint_save){
			sprintf(cline, "write_restart %s", straindata_lcts); lammps_command(lmp,cline); /*opls*/
			//sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_lcts); lammps_command(lmp,cline); /*reaxff*/
			sprintf(cline, "write_restart %s", straindata_time); lammps_command(lmp,cline); /*opls*/
			//sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_time); lammps_command(lmp,cline); /*reaxff*/
		}
		// close down LAMMPS
		delete lmp;

		/*hcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;*/

		// Creating LAMMPS instance
		sprintf(lmparg[4], "%s/log.%s_homogenization", qpreplogloc.c_str(), cellmat.c_str());
		lmp = new LAMMPS(nargs,lmparg,world_communicator);

		//sprintf(cline, "variable locf string %s", locff); lammps_command(lmp,cline); /*reaxff*/
        sprintf(cline, "variable loco string %s", qpreplogloc.c_str()); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_temperature); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.set.lammps");
		lammps_file(lmp,cfile);

		sprintf(mfile, "%s", initdata);
		//sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline); /*reaxff*/

		sprintf(mfile, "%s", straindata_last);
		//sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", mfile); lammps_command(lmp,cline); /*reaxff*/
		sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline); /*opls*/

		sprintf(cline, "variable dts equal %f", md_timestep_length); lammps_command(lmp,cline);

		if(output_homog){
			// Setting dumping of atom positions for post analysis of the MD simulation
			// DO NOT USE CUSTOM DUMP: WRONG ATOM POSITIONS...
			sprintf(cline, "dump atom_dump all atom %d %s", 1, homogdata_time); lammps_command(lmp,cline);
		}

		// Compute the secant stiffness tensor at the given stress/strain state
		sprintf(cline, "variable locbe string %s/%s", scriptsloc.c_str(), "ELASTIC");
		lammps_command(lmp,cline);

		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", md_nsteps_sample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", md_nsteps_sample); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress tensor
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "ELASTIC/in.elastic.lammps");
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
				loc_rep_stress[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*(-1.0)*1.01325e+05;
			}

		if(output_homog){
			// Unetting dumping of atom positions
			sprintf(cline, "undump atom_dump"); lammps_command(lmp,cline);
		}

		// close down LAMMPS
		delete lmp;

		// Cleaning initial offset of stresses
		loc_rep_stress -= init_rep_stress;
	}



	template <int dim>
	void MDProblem<dim>::run (std::string cid, std::string 	tid, std::string cmat,
							  std::string slocout, std::string slocres, std::string llochom,
							  std::string qplogloc, std::string scrloc, std::string mslocout,
							  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
							  double mdss, bool outhom, bool checksav)
	{
		hcout << "Number of processes assigned: " << n_world_processes << std::endl;

		cellid = cid;
		timeid = tid;
		cellmat = cmat;

		statelocout = slocout;
		statelocres = slocres;
		loglochom = llochom;
		qpreplogloc = scrloc;
		scriptsloc = mslocout;
		macrostatelocout = mslocout;

		repl = rep;

		md_timestep_length = mdts;
		md_temperature = mdtem;
		md_nsteps_sample = mdnss;
		md_strain_rate = mdss;

		output_homog = outhom;
		checkpoint_save = checksav;

		char filename[1024];

		// THE NEXT TWO FILES SHOULD BE LOADED IN REPLICADATA INSTEAD
		// Arguments of the secant stiffness computation
		sprintf(filename, "%s/init.%s_%d.stress", macrostatelocout.c_str(), cellmat, repl);
		read_tensor<dim>(filename, init_rep_stress);

		// Providing initial box dimension to adjust the strain tensor
		sprintf(filename, "%s/init.%s_%d.length", macrostatelocout.c_str(), cellmat, repl);
		read_tensor<dim>(filename, init_rep_length);

		// Argument of the MD simulation: strain to apply
		sprintf(filename, "%s/last.%s.%d.upstrain", macrostatelocout.c_str(), cellid, repl);
		read_tensor<dim>(filename, loc_rep_strain);

		// Then the lammps function instanciates lammps, starting from an initial
		// microstructure and applying the complete new_strain or starting from
		// the microstructure at the old_strain and applying the difference between
		// the new_ and _old_strains, returns the new_stress state.
		lammps_straining();

		if(this_world_process == 0)
		{
			std::cout << " \t" << cellid <<"-"<< repl << " \t" << std::flush;

			/*sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout.c_str(), cellid, repl);
			write_tensor<dim>(filename, loc_rep_stiffness);*/

			sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout.c_str(), cellid, repl);
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

		if(argc!=17){
			std::cerr << "Wrong number of arguments, expected: "
					  << "'./single_md cellid timeid cellmat statelocout statelocres"
					  << "loglochom qpreplogloc scriptsloc macrostatelocout repl"
					  << "md_timestep_length md_temperature md_nsteps_sample md_strain_rate"
					  << "output_homog checkpoint_save'"
					  << ", but argc is " << argc << std::endl;
			exit(1);
		}

		std::string cellid = argv[1];
		std::string timeid = argv[2];
		std::string cellmat = argv[3];

		std::string statelocout = argv[4];
		std::string statelocres = argv[5];
		std::string loglochom = argv[6];
		std::string qpreplogloc = argv[7];
		std::string scriptsloc = argv[8];
		std::string macrostatelocout = argv[9];

		unsigned int repl = std::stoi(argv[10]);

		double md_timestep_length = std::stod(argv[11]);
		double md_temperature = std::stod(argv[12]);
		unsigned int md_nsteps_sample = std::stoi(argv[13]);
		double md_strain_rate = std::stod(argv[14]);

		bool output_homog = std::stoi(argv[15]);
		bool checkpoint_save = std::stoi(argv[16]);

		MDProblem<3> md_problem ();

		md_problem.run(cellid, timeid, cellmat, statelocout, statelocres, loglochom,
					   qpreplogloc, scriptsloc, macrostatelocout, repl, md_timestep_length,
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
