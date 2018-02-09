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

namespace HMM
{
	using namespace dealii;
	using namespace LAMMPS_NS;

	template <int dim>
	struct PointHistory
	{
		// History
		SymmetricTensor<2,dim> old_stress;
		SymmetricTensor<2,dim> new_stress;
		SymmetricTensor<2,dim> inc_stress;
		SymmetricTensor<4,dim> old_stiff;
		SymmetricTensor<4,dim> new_stiff;
		SymmetricTensor<2,dim> old_strain;
		SymmetricTensor<2,dim> new_strain;
		SymmetricTensor<2,dim> inc_strain;
		SymmetricTensor<2,dim> upd_strain;
		SymmetricTensor<2,dim> newton_strain;
		bool to_be_updated;

		// Characteristics
		double rho;
		std::string mat;
		Tensor<1,dim> nvec;
		Tensor<2,dim> rotam;
	};

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
	inline
	SymmetricTensor<2,dim>
	rotate_tensor (const SymmetricTensor<2,dim> &tensor,
			const Tensor<2,dim> &rotam)
	{
		SymmetricTensor<2,dim> stmp;

		Tensor<2,dim> tmp;

		Tensor<2,dim> tmp_tensor = tensor;

		tmp = rotam*tmp_tensor*transpose(rotam);

		stmp = 0.5*(tmp + transpose(tmp));

		return stmp;
	}

	template <int dim>
	inline
	SymmetricTensor<4,dim>
	rotate_tensor (const SymmetricTensor<4,dim> &tensor,
			const Tensor<2,dim> &rotam)
	{
		SymmetricTensor<4,dim> tmp;
		tmp = 0;

		// Loop over the indices of the SymmetricTensor (upper "triangle" only)
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				for(unsigned int s=0;s<dim;s++)
					for(unsigned int t=s;t<dim;t++)
					{
						for(unsigned int m=0;m<dim;m++)
							for(unsigned int n=0;n<dim;n++)
								for(unsigned int p=0;p<dim;p++)
									for(unsigned int r=0;r<dim;r++)
										tmp[k][l][s][t] +=
												tensor[m][n][p][r]
												* rotam[k][m] * rotam[l][n]
												* rotam[s][p] * rotam[t][r];
					}

		return tmp;
	}

	template <int dim>
	inline
	SymmetricTensor<2,dim>
	get_strain (const FEValues<dim> &fe_values,
			const unsigned int   shape_func,
			const unsigned int   q_point)
			{
		SymmetricTensor<2,dim> tmp;

		for (unsigned int i=0; i<dim; ++i)
			tmp[i][i] = fe_values.shape_grad_component (shape_func,q_point,i)[i];

		for (unsigned int i=0; i<dim; ++i)
			for (unsigned int j=i+1; j<dim; ++j)
				tmp[i][j]
					   = (fe_values.shape_grad_component (shape_func,q_point,i)[j] +
							   fe_values.shape_grad_component (shape_func,q_point,j)[i]) / 2;

		return tmp;
			}

	template <int dim>
	inline
	SymmetricTensor<2,dim>
	get_strain (const std::vector<Tensor<1,dim> > &grad)
	{
		Assert (grad.size() == dim, ExcInternalError());

		SymmetricTensor<2,dim> strain;
		for (unsigned int i=0; i<dim; ++i)
			strain[i][i] = grad[i][i];

		for (unsigned int i=0; i<dim; ++i)
			for (unsigned int j=i+1; j<dim; ++j)
				strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

		return strain;
	}


	Tensor<2,2>
	get_rotation_matrix (const std::vector<Tensor<1,2> > &grad_u)
	{
		const double curl = (grad_u[1][0] - grad_u[0][1]);

		const double angle = std::atan (curl);

		const double t[2][2] = {{ cos(angle), sin(angle) },
				{-sin(angle), cos(angle) }
		};
		return Tensor<2,2>(t);
	}


	Tensor<2,3>
	get_rotation_matrix (const std::vector<Tensor<1,3> > &grad_u)
	{
		const Point<3> curl (grad_u[2][1] - grad_u[1][2],
				grad_u[0][2] - grad_u[2][0],
				grad_u[1][0] - grad_u[0][1]);

		const double tan_angle = std::sqrt(curl*curl);
		const double angle = std::atan (tan_angle);

		if (angle < 1e-9)
		{
			static const double rotation[3][3]
											= {{ 1, 0, 0}, { 0, 1, 0 }, { 0, 0, 1 } };
			static const Tensor<2,3> rot(rotation);
			return rot;
		}

		const double c = std::cos(angle);
		const double s = std::sin(angle);
		const double t = 1-c;

		const Point<3> axis = curl/tan_angle;
		const double rotation[3][3]
								 = {{
										 t *axis[0] *axis[0]+c,
										 t *axis[0] *axis[1]+s *axis[2],
										 t *axis[0] *axis[2]-s *axis[1]
								 },
										 {
												 t *axis[0] *axis[1]-s *axis[2],
												 t *axis[1] *axis[1]+c,
												 t *axis[1] *axis[2]+s *axis[0]
										 },
										 {
												 t *axis[0] *axis[2]+s *axis[1],
												 t *axis[1] *axis[1]-s *axis[0],
												 t *axis[2] *axis[2]+c
										 }
		};
		return Tensor<2,3>(rotation);
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
					   char* statelocin,
					   char* statelocout,
					   char* logloc,
					   std::string mdt,
					   unsigned int repl)
	{
		// Is this initialization?
		bool init = true;

		// Timestep length in fs
		double dts = 2.0;
		// Number of timesteps factor
		int nsinit = 10000;
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
			char* statelocout,
			char* logloc,
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
		double strain_rate = 1.0e-5; // in fs^(-1)
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
		sprintf(mdstate, "%s_%d.bin", mdt.c_str(), repl);
		char initdata[1024];
		sprintf(initdata, "init.%s", mdstate);

		char atomstate[1024];
		sprintf(atomstate, "%s_%d.atom", mdt.c_str(), repl);

		char replogloc[1024];
		sprintf(replogloc, "%s/R%d", logloc, repl);
		mkdir(replogloc, ACCESSPERMS);

		char qpreplogloc[1024];
		sprintf(qpreplogloc, "%s/%s.%s", replogloc, timeid, cellid);
		mkdir(qpreplogloc, ACCESSPERMS);

		char straindata[1024];
		sprintf(straindata, "%s.%s.%s", timeid, cellid, mdstate);
		char straindata_last[1024];
		sprintf(straindata_last, "last.%s.%s", cellid, mdstate);

		char atomdata_last[1024];
		sprintf(atomdata_last, "last.%s.%s", cellid, atomstate);

		char cline[1024];
		char cfile[1024];
		char mfile[1024];

		// Compute from the initial state (true) or the previous state (false)
		bool compute_finit = false;

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
		sprintf(cline, "dump atom_dump all atom %d %s/%s", ntsdump, statelocout, atomdata_last); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps");
		lammps_file(lmp,cfile);

		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Compute current state data...       " << std::endl;*/
		// Set the state of the testing box at the beginning of the simulation
		// (either from initial end state or from previous testing end state).
		if(compute_finit)
		{
			// Use the initial state if history path in the phases space is to be
			// discarded
			/*if (me == 0) std::cout << "               "
					<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
					<< "   ... from init state data...       " << std::endl;*/
			sprintf(mfile, "%s/%s", statelocout, initdata);
		}
		else
		{
			// Check if a previous state has already been computed specifically for
			// this quadrature point, otherwise use the initial state (which is the
			// last state of this quadrature point)
			/*if (me == 0) std::cout << "               "
					<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
					<< "   ... from previous state data...   " << std::flush;*/
			sprintf(mfile, "%s/%s", statelocout, straindata_last);
			std::ifstream ifile(mfile);
			if (ifile.good()){
				/*if (me == 0) std::cout << "  specifically computed." << std::endl;*/
				ifile.close();

				sprintf(cline, "print 'specifically computed'"); lammps_command(lmp,cline);
			}
			else{
				/*if (me == 0) std::cout << "  initially computed." << std::endl;*/
				sprintf(mfile, "%s/%s", statelocout, initdata);

				sprintf(cline, "print 'initially computed'"); lammps_command(lmp,cline);
			}
		}
		std::ifstream ifile(mfile);
		if (!ifile.good()){
			/*if (me == 0) std::cout << "               "
					<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
					<< "Unable to open beginning state file to read" << std::endl;*/
		}
		else ifile.close();

		sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

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
		sprintf(cline, "write_restart %s/%s", statelocout, straindata_last); lammps_command(lmp,cline);


		/*if (me == 0) std::cout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;*/

		// Compute the secant stiffness tensor at the given stress/strain state
		lammps_homogenization<dim>(lmp, location, stress, stiffness, init);

		// Cleaning initial offset of stresses
		stress -= init_stress;

		// Save data to specific file for this quadrature point
		// At the end of the homogenization the state after sampling the current stress is reread to prepare this write
		//sprintf(cline, "write_restart %s/%s", statelocout, straindata_last); lammps_command(lmp,cline);

		// close down LAMMPS
		delete lmp;
	}



	template <int dim>
	class BodyForce :  public Function<dim>
	{
	public:
		BodyForce ();

		virtual
		void
		vector_value (const Point<dim> &p,
				Vector<double>   &values) const;

		virtual
		void
		vector_value_list (const std::vector<Point<dim> > &points,
				std::vector<Vector<double> >   &value_list) const;
	};


	template <int dim>
	BodyForce<dim>::BodyForce ()
	:
	Function<dim> (dim)
	{}


	template <int dim>
	inline
	void
	BodyForce<dim>::vector_value (const Point<dim> &/*p*/,
			Vector<double>   &values) const
	{
		Assert (values.size() == dim,
				ExcDimensionMismatch (values.size(), dim));

		const double g   = 9.81;

		values = 0;
		values(dim-1) = -g * 0.0;
	}



	template <int dim>
	void
	BodyForce<dim>::vector_value_list (const std::vector<Point<dim> > &points,
			std::vector<Vector<double> >   &value_list) const
	{
		const unsigned int n_points = points.size();

		Assert (value_list.size() == n_points,
				ExcDimensionMismatch (value_list.size(), n_points));

		for (unsigned int p=0; p<n_points; ++p)
			BodyForce<dim>::vector_value (points[p],
					value_list[p]);
	}



	template <int dim>
	class FEProblem
	{
	public:
		FEProblem (MPI_Comm dcomm, int pcolor,
				char* mslocin, char* mslocout, char* mslocres, char* mlogloc,
				std::vector<std::string> mdtype);
		~FEProblem ();

		void restart_system (char* nanostatelocin, char* nanostatelocout, unsigned int nrepl);
		void restart_save (const double present_time, char* nanostatelocout, char* nanostatelocres, unsigned int nrepl) const;

		void make_grid ();
		void setup_system ();
		void set_boundary_values ();
		double assemble_system ();
		void solve_linear_problem_CG ();
		void solve_linear_problem_GMRES ();
		void solve_linear_problem_BiCGStab ();
		void solve_linear_problem_direct ();
		void error_estimation ();
		double determine_step_length () const;
		void move_mesh ();

		void generate_nanostructure();
		std::vector<Vector<double> > get_microstructure ();
		void assign_microstructure (typename DoFHandler<dim>::active_cell_iterator cell, std::vector<Vector<double> > flakes_data,
				std::string &mat, Tensor<2,dim> &rotam);
		void assign_flakes (Point<dim> cpos, std::vector<Vector<double> > flakes_data,
				std::string &mat, Tensor<2,dim> &rotam, double thick_cell);
		void setup_quadrature_point_history ();

		void update_strain_quadrature_point_history
		(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no, const bool updated_stiffnesses);
		void update_stress_quadrature_point_history
		(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no);

		void output_lhistory (const double present_time);
		void output_specific (const double present_time, const int timestep_no, unsigned int nrepl, char* nanostatelocout, char* nanostatelocoutsi);
		void output_visualisation (const double present_time, const int timestep_no);
		void output_results (const double present_time, const int timestep_no, unsigned int nrepl, char* nanostatelocout, char* nanostatelocoutsi);
		void clean_transfer(unsigned int nrepl);

		Vector<double>  compute_internal_forces () const;

		Vector<double> 		     			newton_update_displacement;
		Vector<double> 		     			incremental_displacement;
		Vector<double> 		     			displacement;

	private:
		MPI_Comm 							FE_communicator;
		int 								n_FE_processes;
		int 								this_FE_process;
		int 								FE_pcolor;

		ConditionalOStream 					dcout;

		parallel::shared::Triangulation<dim> triangulation;
		DoFHandler<dim>      				dof_handler;

		FESystem<dim>        				fe;
		const QGauss<dim>   				quadrature_formula;

		ConstraintMatrix     				hanging_node_constraints;
		std::vector<PointHistory<dim> > 	quadrature_point_history;

		PETScWrappers::MPI::SparseMatrix	system_matrix;
		PETScWrappers::MPI::Vector      	system_rhs;

		Vector<float> 						error_per_cell;

		std::vector<types::global_dof_index> local_dofs_per_process;
		IndexSet 							locally_owned_dofs;
		IndexSet 							locally_relevant_dofs;
		unsigned int 						n_local_cells;

		double 								inc_dsupport;
		std::vector<bool> 					topsupport_boundary_dofs;
		std::vector<bool> 					botsupport_boundary_dofs;

		double 								ll;
		double 								hh;
		double 								bb;

		std::vector<std::string> 			mattype;

		char*                               macrostatelocin;
		char*                               macrostatelocout;
		char*                               macrostatelocres;
		char*                               macrologloc;

		std::vector<unsigned int> 			lcis;
		std::vector<unsigned int> 			lcga;
	};



	template <int dim>
	FEProblem<dim>::FEProblem (MPI_Comm dcomm, int pcolor,
			char* mslocin, char* mslocout, char* mslocres, char* mlogloc,
			std::vector<std::string> mdtype)
	:
		FE_communicator (dcomm),
		n_FE_processes (Utilities::MPI::n_mpi_processes(FE_communicator)),
		this_FE_process (Utilities::MPI::this_mpi_process(FE_communicator)),
		FE_pcolor (pcolor),
		dcout (std::cout,(this_FE_process == 0)),
		triangulation(FE_communicator),
		dof_handler (triangulation),
		fe (FE_Q<dim>(1), dim),
		quadrature_formula (2),
		mattype(mdtype),
		macrostatelocin (mslocin),
		macrostatelocout (mslocout),
		macrostatelocres (mslocres),
		macrologloc (mlogloc)
	{}



	template <int dim>
	FEProblem<dim>::~FEProblem ()
	{
		dof_handler.clear ();
	}



	template <int dim>
	std::vector<Vector<double> > FEProblem<dim>::get_microstructure ()
	{
		// Generation of nanostructure based on size, weight ratio
		//generate_nanostructure();

		// Load flakes data (center position, angles, density)
		unsigned int npoints = 0;
		unsigned int nfchar = 0;
		std::vector<Vector<double> > structure_data (npoints, Vector<double>(nfchar));

		char filename[1024];
		sprintf(filename, "%s/structure_data.csv", macrostatelocin);

		std::ifstream ifile;
		ifile.open (filename);

		if (ifile.is_open())
		{
			std::string iline, ival;

			if(getline(ifile, iline)){
				std::istringstream iss(iline);
				if(getline(iss, ival, ',')) npoints = std::stoi(ival);
				if(getline(iss, ival, ',')) nfchar = std::stoi(ival);
			}
			dcout << "Nboxes " << npoints << " - Nchar " << nfchar << std::endl;

			dcout << "Char names: " << std::flush;
			if(getline(ifile, iline)){
				std::istringstream iss(iline);
				for(unsigned int k=0;k<nfchar;k++){
					getline(iss, ival, ',');
					dcout << ival << " " << std::flush;
				}
			}
			dcout << std::endl;

			structure_data.resize(npoints, Vector<double>(nfchar));
			for(unsigned int n=0;n<npoints;n++)
				if(getline(ifile, iline)){
					dcout << "box: " << n << std::flush;
					std::istringstream iss(iline);
					for(unsigned int k=0;k<nfchar;k++){
						getline(iss, ival, ',');
						structure_data[n][k] = std::stof(ival);
						dcout << " - " << structure_data[n][k] << std::flush;
					}
					dcout << std::endl;
				}

			ifile.close();
		}
		else{
			dcout << "Unable to open" << filename << " to read it" << std::endl;
			dcout << "No microstructure loaded!!" << std::endl;
		}

		return structure_data;
	}




	template <int dim>
	void FEProblem<dim>::assign_microstructure (typename DoFHandler<dim>::active_cell_iterator cell, std::vector<Vector<double> > structure_data,
			std::string &mat, Tensor<2,dim> &rotam)
	{
		// Number of flakes
		unsigned int nboxes=structure_data.size();

		// Filling identity matrix
		Tensor<2,dim> idmat;
		idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;

		// Standard properties of cell (pure epoxy)
		mat = mattype[0];

		// Default orientation of cell
		rotam = idmat;

		// Check if the cell contains a graphene flake (composite)
		for(unsigned int n=0;n<nboxes;n++){
			// Load flake center
			Point<dim> fpos (structure_data[n][1],structure_data[n][2],structure_data[n][3]);

			// Load flake normal vector
			Tensor<1,dim> nglo; nglo[0]=structure_data[n][4]; nglo[1]=structure_data[n][5]; nglo[2]=structure_data[n][6];

			if(cell->point_inside(fpos)){

				// Setting composite box material
				for (int imat=1; imat<int(mattype.size()); imat++)
					if(imat == int(structure_data[n][0]))
						mat = mattype[imat];

				//std::cout << " box number: " << n << " is in cell " << cell->active_cell_index()
				//  		  << " of material " << mat << std::endl;

				// Decalaration variables rotation matrix computation
				Tensor<1,dim> nloc;
				double ccos;
				Tensor<2,dim> skew_rot;

				// Write the rotation tensor to the MD box flakes referential (0,1,0)
				nloc[0]=0.0; nloc[1]=1.0; nloc[2]=0.0;

				// Compute the scalar product of the local and global vectors
				ccos = scalar_product(nglo, nloc);

				// Filling the skew-symmetric cross product matrix (a^Tb-b^Ta)
				for (unsigned int i=0; i<dim; ++i)
					for (unsigned int j=0; j<dim; ++j)
						skew_rot[i][j] = nglo[j]*nloc[i] - nglo[i]*nloc[j];

				// Assembling the rotation matrix
				rotam = idmat + skew_rot + (1/(1+ccos))*skew_rot*skew_rot;

				// Stop the for loop since a cell can only be in one flake at a time...
				break;
			}
		}
	}




	template <int dim>
	void FEProblem<dim>::assign_flakes (Point<dim> cpos, std::vector<Vector<double> > flakes_data,
			std::string &mat, Tensor<2,dim> &rotam, double thick_cell)
	{
		// Number of flakes
		unsigned int nflakes=flakes_data.size();

		// Filling identity matrix
		Tensor<2,dim> idmat;
		idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;

		// Standard properties of cell (pure epoxy)
		mat = mattype[0];

		// Default orientation of cell
		rotam = idmat;

		// Check if the cell contains a graphene flake (composite)
		for(unsigned int n=0;n<nflakes;n++){
			// Load flake center
			Point<dim> fpos (flakes_data[n][0],flakes_data[n][1],flakes_data[n][2]);

			// Load flake diameter
			double diam_flake = flakes_data[n][3];

			// Load flake normal vector
			Tensor<1,dim> nglo; nglo[0]=flakes_data[n][4]; nglo[1]=flakes_data[n][5]; nglo[2]=flakes_data[n][6];

			// Load flake thickness
			//double thick_flake = flakes_data[n][7];

			// Compute vector from flake center to cell center
			Tensor<1,dim> vcc; vcc[0]=cpos[0]-fpos[0]; vcc[1]=cpos[1]-fpos[1]; vcc[2]=cpos[2]-fpos[2];

			// Compute normal distance from cell center to flake plane
			double ndist = scalar_product(nglo,vcc);

			// Compute in-plane distance from cell center to flake center
			double pdist = sqrt(vcc.norm_square() - ndist*ndist);

			if(fabs(pdist) < diam_flake/2.0 and (/*fabs(ndist)<thick_flake/2.0 or */fabs(ndist)<thick_cell/2.0)){

//				std::cout << " flake number: " << n << " - pdist: " << pdist << " - ndist: " << ndist
//						  << "  --- cell position: " << cpos[0] << " " << cpos[1] << " " << cpos[2] << " " << std::endl;

				// Setting composite box status
				mat = mattype[1];

				// Decalaration variables rotation matrix computation
				Tensor<1,dim> nloc;
				double ccos;
				Tensor<2,dim> skew_rot;

				// Write the rotation tensor to the MD box flakes referential (0,1,0)
				nloc[0]=0.0; nloc[1]=1.0; nloc[2]=0.0;

				// Compute the scalar product of the local and global vectors
				ccos = scalar_product(nglo, nloc);

				// Filling the skew-symmetric cross product matrix (a^Tb-b^Ta)
				for (unsigned int i=0; i<dim; ++i)
					for (unsigned int j=0; j<dim; ++j)
						skew_rot[i][j] = nglo[j]*nloc[i] - nglo[i]*nloc[j];

				// Assembling the rotation matrix
				rotam = idmat + skew_rot + (1/(1+ccos))*skew_rot*skew_rot;

				// For debug...
				//std::cout << "rot " << rotam[0][0] << " " << rotam[0][1] << " " << rotam[0][2] << std::endl;
				//std::cout << "rot " << rotam[1][0] << " " << rotam[1][1] << " " << rotam[1][2] << std::endl;
				//std::cout << "rot " << rotam[2][0] << " " << rotam[2][1] << " " << rotam[2][2] << std::endl;

				// Stop the for loop since a cell can only be in one flake at a time...
				break;
			}
		}
	}




	template <int dim>
	void FEProblem<dim>::setup_quadrature_point_history ()
	{
		triangulation.clear_user_data();
		{
			std::vector<PointHistory<dim> > tmp;
			tmp.swap (quadrature_point_history);
		}
		quadrature_point_history.resize (n_local_cells *
				quadrature_formula.size());

		// Set materials densities
		std::vector<double>	matdensities;
		matdensities.push_back(1000.);
		matdensities.push_back(1200.);
		matdensities.push_back(1200.);

		char filename[1024];

		// Set materials initial stiffness tensors
		std::vector<SymmetricTensor<4,dim> > stiffness_tensors (mattype.size());

		for(unsigned int imd=0;imd<mattype.size();imd++){
			sprintf(filename, "%s/init.%s.stiff", macrostatelocout, mattype[imd].c_str());
			read_tensor<dim>(filename, stiffness_tensors[imd]);

			if(imd==1){
				stiffness_tensors[imd][0][0][0][0] = stiffness_tensors[imd][0][0][0][0]*3.;
				stiffness_tensors[imd][2][2][2][2] = stiffness_tensors[imd][2][2][2][2]*3.;
			}

			if (imd==0)
				if(this_FE_process==0){
					std::cout << "    Imported initial stiffness..." << std::endl;
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][0][0][0], stiffness_tensors[0][0][0][1][1], stiffness_tensors[imd][0][0][2][2], stiffness_tensors[imd][0][0][0][1], stiffness_tensors[imd][0][0][0][2], stiffness_tensors[imd][0][0][1][2]);
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][1][1][0][0], stiffness_tensors[imd][1][1][1][1], stiffness_tensors[imd][1][1][2][2], stiffness_tensors[imd][1][1][0][1], stiffness_tensors[imd][1][1][0][2], stiffness_tensors[imd][1][1][1][2]);
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][2][2][0][0], stiffness_tensors[imd][2][2][1][1], stiffness_tensors[imd][2][2][2][2], stiffness_tensors[imd][2][2][0][1], stiffness_tensors[imd][2][2][0][2], stiffness_tensors[imd][2][2][1][2]);
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][1][0][0], stiffness_tensors[imd][0][1][1][1], stiffness_tensors[imd][0][1][2][2], stiffness_tensors[imd][0][1][0][1], stiffness_tensors[imd][0][1][0][2], stiffness_tensors[imd][0][1][1][2]);
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][2][0][0], stiffness_tensors[imd][0][2][1][1], stiffness_tensors[imd][0][2][2][2], stiffness_tensors[imd][0][2][0][1], stiffness_tensors[imd][0][2][0][2], stiffness_tensors[imd][0][2][1][2]);
					printf("     %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][1][2][0][0], stiffness_tensors[imd][1][2][1][1], stiffness_tensors[imd][1][2][2][2], stiffness_tensors[imd][1][2][0][1], stiffness_tensors[imd][1][2][0][2], stiffness_tensors[imd][1][2][1][2]);
				}

			sprintf(filename, "%s/last.%s.stiff", macrostatelocout, mattype[imd].c_str());
				write_tensor<dim>(filename, stiffness_tensors[imd]);

		}

		// Application of the microstructure on each cell
		unsigned int history_index = 0;
		for (typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active();
				cell != triangulation.end(); ++cell)
			if (cell->is_locally_owned())
			{
				cell->set_user_pointer (&quadrature_point_history[history_index]);
				history_index += quadrature_formula.size();
			}

		Assert (history_index == quadrature_point_history.size(),
				ExcInternalError());

		// Load the microstructure
		std::vector<Vector<double> > structure_data;
		structure_data = get_microstructure();

		// History data at integration points initialization
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert (local_quadrature_points_history >=
						&quadrature_point_history.front(),
						ExcInternalError());
				Assert (local_quadrature_points_history <
						&quadrature_point_history.back(),
						ExcInternalError());

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					local_quadrature_points_history[q].new_strain = 0;
					local_quadrature_points_history[q].upd_strain = 0;
					local_quadrature_points_history[q].to_be_updated = false;
					local_quadrature_points_history[q].new_stress = 0;

					// Assign microstructure to the current cell (so far, mdtype (?)
					// and rotation from global to local referential of the flake plane
					if (q==0) assign_microstructure(cell, structure_data,
								local_quadrature_points_history[q].mat,
								local_quadrature_points_history[q].rotam);
					else{
						local_quadrature_points_history[q].mat = local_quadrature_points_history[0].mat;
						local_quadrature_points_history[q].rotam = local_quadrature_points_history[0].rotam;
					}

					// Apply stiffness and rotating it from the local sheet orientation (MD) to
					// global orientation (microstructure)
					for (int imat = 0; imat<int(mattype.size()); imat++)
						if(local_quadrature_points_history[q].mat==mattype[imat]){
							local_quadrature_points_history[q].new_stiff =
								rotate_tensor(stiffness_tensors[imat],
									transpose(local_quadrature_points_history[q].rotam));

							// Apply composite density
							local_quadrature_points_history[q].rho = matdensities[imat];
						}
				}

				/*std::cout << "   cell " << cell->active_cell_index()
						  << "   mat "  << local_quadrature_points_history[5].mat
						  << "   dens " << local_quadrature_points_history[6].rho
						  << std::endl;*/
			}

		/*for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert (local_quadrature_points_history >=
						&quadrature_point_history.front(),
						ExcInternalError());
				Assert (local_quadrature_points_history <
						&quadrature_point_history.back(),
						ExcInternalError());

				std::cout << " cell: " << cell->active_cell_index() << " - mat: "
						<< local_quadrature_points_history[0].mat << " "
						<< local_quadrature_points_history[1].mat
						<< std::endl;
			}*/
	}




	template <int dim>
	void FEProblem<dim>::make_grid ()
	{
		ll=0.000050;
		hh=0.000150;
		bb=0.000050;

		char filename[1024];
		sprintf(filename, "%s/mesh.tria", macrostatelocin);

		std::ifstream iss(filename);
		if (iss.is_open()){
			dcout << "    Reuse of existing triangulation... "
				  << "(requires the exact SAME COMMUNICATOR!!)" << std::endl;
			boost::archive::text_iarchive ia(iss, boost::archive::no_header);
			triangulation.load(ia, 0);
		}
		else{
			dcout << "    Creation of triangulation..." << std::endl;
			Point<dim> pp1 (-ll/2.,-hh/2.,-bb/2.);
			Point<dim> pp2 (ll/2.,hh/2.,bb/2.);
			std::vector< unsigned int > reps (dim);
			reps[0] = 10; reps[1] = 30; reps[2] = 10;
			GridGenerator::subdivided_hyper_rectangle(triangulation, reps, pp1, pp2);

			//triangulation.refine_global (1);

			// Saving triangulation, not usefull now and costly...
			/*sprintf(filename, "%s/mesh.tria", macrostatelocout);
			std::ofstream oss(filename);
			boost::archive::text_oarchive oa(oss, boost::archive::no_header);
			triangulation.save(oa, 0);*/
		}

		dcout << "    Number of active cells:       "
				<< triangulation.n_active_cells()
				<< " (by partition:";
		for (int p=0; p<n_FE_processes; ++p)
			dcout << (p==0 ? ' ' : '+')
			<< (GridTools::
					count_cells_with_subdomain_association (triangulation,p));
		dcout << ")" << std::endl;
	}




	template <int dim>
	void FEProblem<dim>::setup_system ()
	{
		dof_handler.distribute_dofs (fe);
		locally_owned_dofs = dof_handler.locally_owned_dofs();
		DoFTools::extract_locally_relevant_dofs (dof_handler,locally_relevant_dofs);

		n_local_cells
		= GridTools::count_cells_with_subdomain_association (triangulation,
				triangulation.locally_owned_subdomain ());
		local_dofs_per_process = dof_handler.n_locally_owned_dofs_per_processor();

		hanging_node_constraints.clear ();
		DoFTools::make_hanging_node_constraints (dof_handler,
				hanging_node_constraints);
		hanging_node_constraints.close ();

		DynamicSparsityPattern sparsity_pattern (locally_relevant_dofs);
		DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern,
				hanging_node_constraints, false);
		SparsityTools::distribute_sparsity_pattern (sparsity_pattern,
				local_dofs_per_process,
				FE_communicator,
				locally_relevant_dofs);

		system_matrix.reinit (locally_owned_dofs,
				locally_owned_dofs,
				sparsity_pattern,
				FE_communicator);
		system_rhs.reinit (locally_owned_dofs, FE_communicator);

		newton_update_displacement.reinit (dof_handler.n_dofs());
		incremental_displacement.reinit (dof_handler.n_dofs());
		displacement.reinit (dof_handler.n_dofs());

		dcout << "    Number of degrees of freedom: "
				<< dof_handler.n_dofs()
				<< " (by partition:";
		for (int p=0; p<n_FE_processes; ++p)
			dcout << (p==0 ? ' ' : '+')
			<< (DoFTools::
					count_dofs_with_subdomain_association (dof_handler,p));
		dcout << ")" << std::endl;
	}




	template <int dim>
	void FEProblem<dim>::update_strain_quadrature_point_history
	(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no, const bool updated_stiffnesses)
	{
		// Create file with qptid to update at timeid
		std::ofstream ofile;
		char update_local_filename[1024];
		sprintf(update_local_filename, "%s/last.%d.qpupdates", macrostatelocout, this_FE_process);
		ofile.open (update_local_filename);

		// Create file with mattype of qptid to update at timeid
		std::ofstream omatfile;
		char mat_update_local_filename[1024];
		sprintf(mat_update_local_filename, "%s/last.%d.matqpupdates", macrostatelocout, this_FE_process);
		omatfile.open (mat_update_local_filename);

		// Preparing requirements for strain update
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		double strain_perturbation = 0.025;

		char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);

		if (newtonstep_no > 0) dcout << "        " << "...checking quadrature points requiring update..." << std::endl;

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				SymmetricTensor<2,dim> newton_strain_tensor, avg_upd_strain_tensor;

				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert (local_quadrature_points_history >=
						&quadrature_point_history.front(),
						ExcInternalError());
				Assert (local_quadrature_points_history <
						&quadrature_point_history.back(),
						ExcInternalError());
				fe_values.reinit (cell);
				fe_values.get_function_gradients (displacement_update,
						displacement_update_grads);

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
					local_quadrature_points_history[q].to_be_updated = false;

				avg_upd_strain_tensor = 0.;

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					local_quadrature_points_history[q].old_strain =
							local_quadrature_points_history[q].new_strain;

					local_quadrature_points_history[q].old_stress =
							local_quadrature_points_history[q].new_stress;

					local_quadrature_points_history[q].old_stiff =
							local_quadrature_points_history[q].new_stiff;

					if (newtonstep_no == 0) local_quadrature_points_history[q].inc_strain = 0.;

					// Strain tensor update
					local_quadrature_points_history[q].newton_strain = get_strain (displacement_update_grads[q]);
					local_quadrature_points_history[q].inc_strain += local_quadrature_points_history[q].newton_strain;
					local_quadrature_points_history[q].new_strain += local_quadrature_points_history[q].newton_strain;
					local_quadrature_points_history[q].upd_strain += local_quadrature_points_history[q].newton_strain;

					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++)
							avg_upd_strain_tensor[k][l] += local_quadrature_points_history[q].upd_strain[k][l];
				}

				for(unsigned int k=0;k<dim;k++)
					for(unsigned int l=k;l<dim;l++)
						avg_upd_strain_tensor[k][l] /= quadrature_formula.size();


				bool cell_to_be_updated = false;
				//if ((cell->active_cell_index() == 240)) // For debug...
				//if (false) // For debug...
				if (newtonstep_no > 0 && !updated_stiffnesses)
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++)
							if (fabs(avg_upd_strain_tensor[k][l]) > strain_perturbation
									&& cell_to_be_updated == false){
								std::cout << "           "
										<< " cell "<< cell->active_cell_index()
										<< " strain component " << k << l
										<< " value " << avg_upd_strain_tensor[k][l] << std::endl;

								cell_to_be_updated = true;
								for (unsigned int qc=0; qc<quadrature_formula.size(); ++qc)
									local_quadrature_points_history[qc].to_be_updated = true;

								// Write strains since last update in a file named ./macrostate_storage/last.cellid-qid.strain
								char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());
								char filename[1024];

								SymmetricTensor<2,dim> rot_avg_upd_strain_tensor;

								if(local_quadrature_points_history[0].mat==mattype[1])
									// Rotation of the strain update tensor wrt to the flake angle
									rot_avg_upd_strain_tensor =
											rotate_tensor(avg_upd_strain_tensor, local_quadrature_points_history[0].rotam);
								else rot_avg_upd_strain_tensor = avg_upd_strain_tensor;

								sprintf(filename, "%s/last.%s.upstrain", macrostatelocout, cell_id);
								write_tensor<dim>(filename, rot_avg_upd_strain_tensor);

								ofile << cell_id << std::endl;
								omatfile << local_quadrature_points_history[0].mat << std::endl;
							}
			}
		ofile.close();
		MPI_Barrier(FE_communicator);

		// Gathering in a single file all the quadrature points to be updated...
		// Might be worth replacing indivual local file writings by a parallel vector of string
		// and globalizing this vector before this final writing step.
		std::ifstream infile;
		std::ofstream outfile;
		std::string iline;
		if (this_FE_process == 0){
			char update_filename[1024];

			sprintf(update_filename, "%s/last.qpupdates", macrostatelocout);
			outfile.open (update_filename);
			for (int ip=0; ip<n_FE_processes; ip++){
				sprintf(update_local_filename, "%s/last.%d.qpupdates", macrostatelocout, ip);
				infile.open (update_local_filename);
				while (getline(infile, iline)) outfile << iline << std::endl;
				infile.close();
			}
			outfile.close();

			sprintf(update_filename, "%s/last.matqpupdates", macrostatelocout);
			outfile.open (update_filename);
			for (int ip=0; ip<n_FE_processes; ip++){
				sprintf(update_local_filename, "%s/last.%d.matqpupdates", macrostatelocout, ip);
				infile.open (update_local_filename);
				while (getline(infile, iline)) outfile << iline << std::endl;
				infile.close();
			}
			outfile.close();

			char alltime_update_filename[1024];
			sprintf(alltime_update_filename, "%s/alltime_cellupdates.dat", macrologloc);
			outfile.open (alltime_update_filename, std::ofstream::app);
			if(timestep_no==1 && newtonstep_no==1) outfile << "timestep_no,newtonstep_no,cell" << std::endl;
			infile.open (update_filename);
			while (getline(infile, iline)) outfile << timestep_no << "," << newtonstep_no << "," << iline << std::endl;
			infile.close();
			outfile.close();

			// Save quadrature point updates history for later checking...
//			sprintf(update_filename, "%s/last.qpupdates", macrostatelocout);
//		    std::ifstream  macroin(update_filename, std::ios::binary);
//		    sprintf(update_filename, "%s/%s.qpupdates", macrostatelocout, time_id);
//		    std::ofstream  macroout(update_filename,   std::ios::binary);
//		    macroout << macroin.rdbuf();
//		    macroin.close();
//		    macroout.close();
		}
	}




	template <int dim>
	void FEProblem<dim>::update_stress_quadrature_point_history
	(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no)
	{
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);

		// Retrieving all quadrature points computation and storing them in the
		// quadrature_points_history structure
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				SymmetricTensor<2,dim> avg_upd_strain_tensor;
				//SymmetricTensor<2,dim> avg_stress_tensor;

				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert (local_quadrature_points_history >=
						&quadrature_point_history.front(),
						ExcInternalError());
				Assert (local_quadrature_points_history <
						&quadrature_point_history.back(),
						ExcInternalError());
				fe_values.reinit (cell);
				fe_values.get_function_gradients (displacement_update,
						displacement_update_grads);

				// Restore the new stiffness tensors from ./macroscale_state/out/last.cellid-qid.stiff
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());
				char filename[1024];

				avg_upd_strain_tensor = 0.;
				//avg_stress_tensor = 0.;

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					// For debug...
					/*if (local_quadrature_points_history[q].mat==mattype[1]
							and q==0){

						SymmetricTensor<2,dim> tmp_stress = local_quadrature_points_history[q].new_stress;

						std::cout << "ori " << tmp_stress[0][0] << " " << tmp_stress[0][1] << " " << tmp_stress[0][2] << std::endl;
						std::cout << "ori " << tmp_stress[1][0] << " " << tmp_stress[1][1] << " " << tmp_stress[1][2] << std::endl;
						std::cout << "ori " << tmp_stress[2][0] << " " << tmp_stress[2][1] << " " << tmp_stress[2][2] << std::endl;

						SymmetricTensor<2,dim> rot_stress;
						rot_stress =
							rotate_tensor(tmp_stress, transpose(local_quadrature_points_history[q].rotam));

						std::cout << "rot " << rot_stress[0][0] << " " << rot_stress[0][1] << " " << rot_stress[0][2] << std::endl;
						std::cout << "rot " << rot_stress[1][0] << " " << rot_stress[1][1] << " " << rot_stress[1][2] << std::endl;
						std::cout << "rot " << rot_stress[2][0] << " " << rot_stress[2][1] << " " << rot_stress[2][2] << std::endl;
					}*/

					if (newtonstep_no == 0) local_quadrature_points_history[q].inc_stress = 0.;

					if (local_quadrature_points_history[q].to_be_updated){

						// Updating stiffness tensor
						/*SymmetricTensor<4,dim> stmp_stiff;
						sprintf(filename, "%s/last.%s.stiff", macrostatelocout, cell_id);
						read_tensor<dim>(filename, stmp_stiff);

						if(local_quadrature_points_history[q].mat==mattype[1])
							// Rotate the output stiffness wrt the flake angles
							local_quadrature_points_history[q].new_stiff =
									rotate_tensor(stmp_stiff, transpose(local_quadrature_points_history[q].rotam));

						else local_quadrature_points_history[q].new_stiff = stmp_stiff;*/

						// Updating stress tensor
						SymmetricTensor<2,dim> stmp_stress;
						sprintf(filename, "%s/last.%s.stress", macrostatelocout, cell_id);
						read_tensor<dim>(filename, stmp_stress);

						if (local_quadrature_points_history[q].mat==mattype[1]){
							// Rotate the output stress wrt the flake angles
							local_quadrature_points_history[q].new_stress =
									rotate_tensor(stmp_stress, transpose(local_quadrature_points_history[q].rotam));
						}
						else local_quadrature_points_history[q].new_stress = stmp_stress;

						// Resetting the update strain tensor
						local_quadrature_points_history[q].upd_strain = 0;
					}
					else{
						// Tangent stiffness computation of the new stress tensor and the stress increment tensor
						local_quadrature_points_history[q].inc_stress +=
							local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].newton_strain;

						local_quadrature_points_history[q].new_stress +=
							local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].newton_strain;
					}

					// Secant stiffness computation of the new stress tensor
					//local_quadrature_points_history[q].new_stress =
					//		local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].new_strain;

					// Write stress tensor for each gauss point
					/*sprintf(filename, "%s/last.%s-%d.stress", macrostatelocout, cell_id,q);
					write_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);*/

					// Averaging upd_strain over cell
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++)
							avg_upd_strain_tensor[k][l] += local_quadrature_points_history[q].upd_strain[k][l];

					/*for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++)
							avg_stress_tensor[k][l] += local_quadrature_points_history[q].new_stress[k][l];*/

					// Apply rotation of the sample to the new state tensors.
					// Only needed if the mesh is modified...
					/*const Tensor<2,dim> rotation
					= get_rotation_matrix (displacement_update_grads[q]);

					const SymmetricTensor<2,dim> rotated_new_stress
					= symmetrize(transpose(rotation) *
							static_cast<Tensor<2,dim> >
					(local_quadrature_points_history[q].new_stress) *
					rotation);

					const SymmetricTensor<2,dim> rotated_new_strain
					= symmetrize(transpose(rotation) *
							static_cast<Tensor<2,dim> >
					(local_quadrature_points_history[q].new_strain) *
					rotation);

					const SymmetricTensor<2,dim> rotated_upd_strain
					= symmetrize(transpose(rotation) *
							static_cast<Tensor<2,dim> >
					(local_quadrature_points_history[q].upd_strain) *
					rotation);

					local_quadrature_points_history[q].new_stress
					= rotated_new_stress;
					local_quadrature_points_history[q].new_strain
					= rotated_new_strain;
					local_quadrature_points_history[q].upd_strain
					= rotated_upd_strain;*/
				}
			}
	}


	// Might want to restructure this function to avoid repetitions
	// with boundary conditions correction performed at the end of the
	// assemble_system() function
	template <int dim>
	void FEProblem<dim>::set_boundary_values()
	{
		inc_dsupport = +0.0000005;

		FEValuesExtractors::Scalar x_component (dim-3);
		FEValuesExtractors::Scalar y_component (dim-2);
		FEValuesExtractors::Scalar z_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;


		topsupport_boundary_dofs.resize(dof_handler.n_dofs());
		botsupport_boundary_dofs.resize(dof_handler.n_dofs());

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

		for ( ; cell != endc; ++cell) {
			double eps = (cell->minimum_vertex_distance());
			for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v) {

				unsigned int component;
				double value;

				for (unsigned int c = 0; c < dim; ++c) {
					botsupport_boundary_dofs[cell->vertex_dof_index (v, c)] = false;
					topsupport_boundary_dofs[cell->vertex_dof_index (v, c)] = false;
				}

				if (fabs(cell->vertex(v)(1) - -hh/2.) < eps/3.)
				{
					if (fabs(cell->vertex(v)(0) - 0.) < eps/3.
							and fabs(cell->vertex(v)(2) - 0.) < eps/3.){
						value = 0.;
						component = 0;
						botsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->vertex_dof_index (v, component), value));

						value = 0.;
						component = 2;
						botsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->vertex_dof_index (v, component), value));
					}

					value = 0.;
					component = 1;
					botsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));

//					dcout << "bot support type"
//						  << " -- dof id: " << cell->vertex_dof_index (v, component)
//						  << " -- position: " << cell->vertex(v)(0) << " - " << cell->vertex(v)(1) << " - " << cell->vertex(v)(2) << " - " << std::endl;
				}


				if (fabs(cell->vertex(v)(1) - +hh/2.) < eps/3.)
				{
					if (fabs(cell->vertex(v)(0) - 0.) < eps/3.
							and fabs(cell->vertex(v)(2) - 0.) < eps/3.){
						value = 0.;
						component = 0;
						topsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->vertex_dof_index (v, component), value));

						value = 0.;
						component = 2;
						topsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->vertex_dof_index (v, component), value));
					}

					value = inc_dsupport;
					component = 1;
					topsupport_boundary_dofs[cell->vertex_dof_index (v, component)] = true;
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));

//					dcout << "top support type"
//						  << " -- dof id: " << cell->vertex_dof_index (v, component)
//						  << " -- position: " << cell->vertex(v)(0) << " - " << cell->vertex(v)(1) << " - " << cell->vertex(v)(2) << " - " << std::endl;
				}
			}
		}


		for (std::map<types::global_dof_index, double>::const_iterator
				p = boundary_values.begin();
				p != boundary_values.end(); ++p)
			incremental_displacement(p->first) = p->second;
	}



	template <int dim>
	double FEProblem<dim>::assemble_system ()
	{
		double rhs_residual;

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values   | update_gradients |
				update_quadrature_points | update_JxW_values);

		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		const unsigned int   n_q_points    = quadrature_formula.size();

		FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
		Vector<double>       cell_rhs (dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		BodyForce<dim>      body_force;
		std::vector<Vector<double> > body_force_values (n_q_points,
				Vector<double>(dim));

		system_rhs = 0;
		system_matrix = 0;

		for (; cell!=endc; ++cell)
			if (cell->is_locally_owned())
			{
				cell_matrix = 0;
				cell_rhs = 0;

				fe_values.reinit (cell);

				const PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

				// Assembly of stiffness matrix
				for (unsigned int i=0; i<dofs_per_cell; ++i)
					for (unsigned int j=0; j<dofs_per_cell; ++j)
						for (unsigned int q_point=0; q_point<n_q_points;
								++q_point)
						{
							const SymmetricTensor<4,dim> &new_stiff
							= local_quadrature_points_history[q_point].new_stiff;

							const SymmetricTensor<2,dim>
							eps_phi_i = get_strain (fe_values, i, q_point),
							eps_phi_j = get_strain (fe_values, j, q_point);

							cell_matrix(i,j)
							+= (eps_phi_i * new_stiff * eps_phi_j
									*
									fe_values.JxW (q_point));
						}

				body_force.vector_value_list (fe_values.get_quadrature_points(),
						body_force_values);

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					const unsigned int
					component_i = fe.system_to_component_index(i).first;

					for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
					{
						body_force.vector_value_list (fe_values.get_quadrature_points(),
								body_force_values);

						const SymmetricTensor<2,dim> &new_stress
						= local_quadrature_points_history[q_point].new_stress;

						// how to handle body forces?
						cell_rhs(i) += (
								body_force_values[q_point](component_i) *
								local_quadrature_points_history[q_point].rho *
								fe_values.shape_value (i,q_point)
								-
								new_stress *
								get_strain (fe_values,i,q_point))
								*
								fe_values.JxW (q_point);
					}
				}

				cell->get_dof_indices (local_dof_indices);

				hanging_node_constraints
				.distribute_local_to_global(cell_matrix, cell_rhs,
						local_dof_indices,
						system_matrix, system_rhs);
			}

		system_matrix.compress(VectorOperation::add);
		system_rhs.compress(VectorOperation::add);


		FEValuesExtractors::Scalar x_component (dim-3);
		FEValuesExtractors::Scalar y_component (dim-2);
		FEValuesExtractors::Scalar z_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;

		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

		// Apply velocity boundary conditions
		for ( ; cell != endc; ++cell) {
			for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v) {
				unsigned int component;
				double value;

				value = 0.;
				component = 0;
				if (botsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}

				value = 0.;
				component = 1;
				if (botsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}

				value = 0.;
				component = 2;
				if (botsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}

				value = 0.;
				component = 0;
				if (topsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}

				value = 0.;
				component = 1;
				if (topsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}

				value = 0.;
				component = 2;
				if (topsupport_boundary_dofs[cell->vertex_dof_index (v, component)])
				{
					boundary_values.insert(std::pair<types::global_dof_index, double>
					(cell->vertex_dof_index (v, component), value));
				}
			}
		}

		PETScWrappers::MPI::Vector tmp (locally_owned_dofs,FE_communicator);
		MatrixTools::apply_boundary_values (boundary_values,
				system_matrix,
				tmp,
				system_rhs,
				false);
		newton_update_displacement = tmp;

		rhs_residual = system_rhs.l2_norm();
		dcout << "    FE System - norm of rhs is " << rhs_residual
							  << std::endl;

		return rhs_residual;
	}



	template <int dim>
	void FEProblem<dim>::solve_linear_problem_CG ()
	{
		PETScWrappers::MPI::Vector
		distributed_newton_update (locally_owned_dofs,FE_communicator);
		distributed_newton_update = newton_update_displacement;

		// The residual used internally to test solver convergence is
		// not identical to ours, it probably considers preconditionning.
		// Therefore, extra precision is required in the solver proportionnaly
		// to the norm of the system matrix, to reduce sufficiently our residual
		SolverControl       solver_control (dof_handler.n_dofs(),
				1e-03/system_matrix.l1_norm());

		PETScWrappers::SolverCG cg (solver_control,
				FE_communicator);

		// Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
		// not optimal for large scale simulations.
		PETScWrappers::PreconditionJacobi preconditioner(system_matrix);
		cg.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_displacement = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_displacement);

		const double alpha = determine_step_length();
		incremental_displacement.add (alpha, newton_update_displacement);

		dcout << "    FE Solver - norm of newton update is " << newton_update_displacement.l2_norm()
							  << std::endl;
		dcout << "    FE Solver converged in " << solver_control.last_step()
				<< " iterations "
				<< " with value " << solver_control.last_value()
				<<  std::endl;
	}



	template <int dim>
	void FEProblem<dim>::solve_linear_problem_GMRES ()
	{
		PETScWrappers::MPI::Vector
		distributed_newton_update (locally_owned_dofs,FE_communicator);
		distributed_newton_update = newton_update_displacement;

		// The residual used internally to test solver convergence is
		// not identical to ours, it probably considers preconditionning.
		// Therefore, extra precision is required in the solver proportionnaly
		// to the norm of the system matrix, to reduce sufficiently our residual
		SolverControl       solver_control (dof_handler.n_dofs(),
				1e-03/system_matrix.l1_norm());

		PETScWrappers::SolverGMRES gmres (solver_control,
				FE_communicator);

		// Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
		// not optimal for large scale simulations.
		PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
		gmres.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_displacement = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_displacement);

		const double alpha = determine_step_length();
		incremental_displacement.add (alpha, newton_update_displacement);

		dcout << "    FE Solver - norm of newton update is " << newton_update_displacement.l2_norm()
							  << std::endl;
		dcout << "    FE Solver converged in " << solver_control.last_step()
				<< " iterations "
				<< " with value " << solver_control.last_value()
				<<  std::endl;
	}



	template <int dim>
	void FEProblem<dim>::solve_linear_problem_BiCGStab ()
	{
		PETScWrappers::MPI::Vector
		distributed_newton_update (locally_owned_dofs,FE_communicator);
		distributed_newton_update = newton_update_displacement;

		PETScWrappers::PreconditionBoomerAMG preconditioner;
		  {
		    PETScWrappers::PreconditionBoomerAMG::AdditionalData additional_data;
		    additional_data.symmetric_operator = true;

		    preconditioner.initialize(system_matrix, additional_data);
		  }

		// The residual used internally to test solver convergence is
		// not identical to ours, it probably considers preconditionning.
		// Therefore, extra precision is required in the solver proportionnaly
		// to the norm of the system matrix, to reduce sufficiently our residual
		SolverControl       solver_control (dof_handler.n_dofs(),
				1e-12/system_matrix.l1_norm());

		PETScWrappers::SolverBicgstab bicgs (solver_control,
				FE_communicator);

		bicgs.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_displacement = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_displacement);

		const double alpha = determine_step_length();
		incremental_displacement.add (alpha, newton_update_displacement);

		dcout << "    FE Solver - norm of newton update is " << newton_update_displacement.l2_norm()
							  << std::endl;
		dcout << "    FE Solver converged in " << solver_control.last_step()
				<< " iterations "
				<< " with value " << solver_control.last_value()
				<<  std::endl;
	}



	template <int dim>
	void FEProblem<dim>::solve_linear_problem_direct ()
	{
		PETScWrappers::MPI::Vector
		distributed_newton_update (locally_owned_dofs,FE_communicator);
		distributed_newton_update = newton_update_displacement;

		SolverControl       solver_control;

		PETScWrappers::SparseDirectMUMPS solver (solver_control,
				FE_communicator);

		//solver.set_symmetric_mode(false);

		solver.solve (system_matrix, distributed_newton_update, system_rhs);
		//system_inverse.vmult(distributed_newton_update, system_rhs);

		newton_update_displacement = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_displacement);

		const double alpha = determine_step_length();
		incremental_displacement.add (alpha, newton_update_displacement);

		dcout << "    FE Solver - norm of newton update is " << newton_update_displacement.l2_norm()
							  << std::endl;
	}



	template <int dim>
	Vector<double> FEProblem<dim>::compute_internal_forces () const
	{
		PETScWrappers::MPI::Vector residual
		(locally_owned_dofs, FE_communicator);

		residual = 0;

		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values   | update_gradients |
				update_quadrature_points | update_JxW_values);

		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		const unsigned int   n_q_points    = quadrature_formula.size();

		Vector<double>               cell_residual (dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
			if (cell->is_locally_owned())
			{
				cell_residual = 0;
				fe_values.reinit (cell);

				const PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
					{
						const SymmetricTensor<2,dim> &old_stress
						= local_quadrature_points_history[q_point].new_stress;

						cell_residual(i) +=
								(old_stress *
								get_strain (fe_values,i,q_point))
								*
								fe_values.JxW (q_point);
					}
				}

				cell->get_dof_indices (local_dof_indices);
				hanging_node_constraints.distribute_local_to_global
				(cell_residual, local_dof_indices, residual);
			}

		residual.compress(VectorOperation::add);

		Vector<double> local_residual (dof_handler.n_dofs());
		local_residual = residual;

		return local_residual;
	}



	template <int dim>
	void FEProblem<dim>::error_estimation ()
	{
		error_per_cell.reinit (triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate (dof_handler,
				QGauss<dim-1>(2),
				typename FunctionMap<dim>::type(),
				newton_update_displacement,
				error_per_cell,
				ComponentMask(),
				0,
				MultithreadInfo::n_threads(),
				this_FE_process);

		// Not too sure how is stored the vector 'distributed_error_per_cell',
		// it might be worth checking in case this is local, hence using a
		// lot of memory on a single process. This is ok, however it might
		// stupid to keep this vector global because the memory space will
		// be kept used during the whole simulation.
		const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells ();
		PETScWrappers::MPI::Vector
		distributed_error_per_cell (FE_communicator,
				triangulation.n_active_cells(),
				n_local_cells);
		for (unsigned int i=0; i<error_per_cell.size(); ++i)
			if (error_per_cell(i) != 0)
				distributed_error_per_cell(i) = error_per_cell(i);
		distributed_error_per_cell.compress (VectorOperation::insert);

		error_per_cell = distributed_error_per_cell;
	}




	template <int dim>
	double FEProblem<dim>::determine_step_length() const
	{
		return 1.0;
	}




	template <int dim>
	void FEProblem<dim>::move_mesh ()
	{
		dcout << "    Moving mesh..." << std::endl;

		std::vector<bool> vertex_touched (triangulation.n_vertices(),
				false);
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active ();
				cell != dof_handler.end(); ++cell)
			for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
				if (vertex_touched[cell->vertex_index(v)] == false)
				{
					vertex_touched[cell->vertex_index(v)] = true;

					Point<dim> vertex_displacement;
					for (unsigned int d=0; d<dim; ++d)
						vertex_displacement[d]
											= incremental_displacement(cell->vertex_dof_index(v,d));

					cell->vertex(v) += vertex_displacement;
				}
	}



	template <int dim>
	void FEProblem<dim>::output_lhistory (const double present_time)
	{
		char filename[1024];

		// Initialization of the processor local history data file
		sprintf(filename, "%s/pr_%d.lhistory.csv", macrologloc, this_FE_process);
		std::ofstream  lhprocout(filename, std::ios_base::app);
		long cursor_position = lhprocout.tellp();

		if (cursor_position == 0)
		{
			lhprocout << "timestep,cell,qpoint,material";
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					lhprocout << "," << "strain_" << k << l;
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					lhprocout  << "," << "updstrain_" << k << l;
			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
					lhprocout << "," << "stress_" << k << l;
			lhprocout << std::endl;
		}

		// Output of complete local history in a single file per processor
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());

				PointHistory<dim> *local_qp_hist
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

				// Save strain, updstrain, stress history in one file per proc
				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					lhprocout << present_time
							<< "," << cell->active_cell_index()
							<< "," << q
							<< "," << local_qp_hist[q].mat.c_str();
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							lhprocout << "," << std::setprecision(16) << local_qp_hist[q].new_strain[k][l];
						}
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							lhprocout << "," << std::setprecision(16) << local_qp_hist[q].upd_strain[k][l];
						}
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							lhprocout << "," << std::setprecision(16) << local_qp_hist[q].new_stress[k][l];
						}
					lhprocout << std::endl;
				}
			}
		lhprocout.close();
	}



	template <int dim>
	void FEProblem<dim>::output_specific (const double present_time, const int timestep_no, unsigned int nrepl, char* nanostatelocout, char* nanologlocsi)
	{
		// Build lists of cells for output
		if(timestep_no==1){
			dcout << "Cells with detailed output: " << std::endl;
			for (typename DoFHandler<dim>::active_cell_iterator
					cell = dof_handler.begin_active();
					cell != dof_handler.end(); ++cell)
			{
				double eps = (cell->minimum_vertex_distance());

				// Build vector of ids of cells of special interest 'lcis'
				//for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v) {
					if (cell->barycenter()(1) <  (hh)/2. && cell->barycenter()(1) >  -((hh)/2.)
							&& fabs(cell->barycenter()(0) - eps/2.) < eps/3.
							&& fabs(cell->barycenter()(2) - 0.0) < 2.*eps/3.){
						lcis.push_back(cell->active_cell_index());
						dcout << " specific cell: " << cell->active_cell_index() << " y: " << cell->barycenter()(1) << std::endl;
					}
					if (fabs(cell->barycenter()(0) - eps/2.) >= eps/3.
							&& fabs(cell->barycenter()(1) - eps/2.) < eps/3.
							&& fabs(cell->barycenter()(2) - 0.0) < 2.*eps/3.){
						lcis.push_back(cell->active_cell_index());
						dcout << " specific cell: " << cell->active_cell_index() << " x: " << cell->barycenter()(0) << std::endl;
					}
					if (fabs(cell->barycenter()(2) - eps/2.) >= eps/3.
							&& fabs(cell->barycenter()(1) - eps/2.) < eps/3.
							&& fabs(cell->barycenter()(0) - eps/2.) < eps/3.){
						lcis.push_back(cell->active_cell_index());
						dcout << " specific cell: " << cell->active_cell_index() << " z: " << cell->barycenter()(2) << std::endl;
					}
				//}
				// Build vector of ids of cells for measuring gauge displacement 'lcga'
				//for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v) {
					if ((fabs(cell->barycenter()(1) - hh/2.) < 2.*eps/3. || fabs(cell->barycenter()(1) - -hh/2.) < 2.*eps/3.)
						&& fabs(cell->barycenter()(0) - eps/2.) < eps/3.
						&& fabs(cell->barycenter()(2) - eps/2.) < eps/3.)
					{
						lcga.push_back(cell->active_cell_index());
						dcout << " gauge cell: " << cell->active_cell_index() << " y: " << cell->barycenter()(1) << std::endl;
					}
				//}
			}
		}

		// Compute applied force vector
		Vector<double> local_residual (dof_handler.n_dofs());
		local_residual = compute_internal_forces();

		// Compute force under the loading boundary condition
		double aforce = 0.;
		//dcout << "hello Y force ------ " << std::endl;
		for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
			if (topsupport_boundary_dofs[i] == true)
			{
				// For Debug...
				//dcout << "   force on loaded nodes: " << local_residual[i] << std::endl;
				aforce += local_residual[i];
			}

		// Compute displacement of the gauge
		double idisp = 0.;
		double ytop = 0.;
		double ybot = 0.;
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
		{
			for(unsigned int ii=0;ii<lcga.size();ii++){
				if (cell->active_cell_index()==lcga[ii])
				{
					double ypos = cell->vertex(0)(1)+displacement[cell->vertex_dof_index (0, 1)];
					if(ypos > 0.0) ytop = ypos;
					else ybot = ypos;
				}
			}
		}
		idisp = ytop-ybot;
		dcout << "Timestep: " << timestep_no << " - Time: " << present_time << " - Gauge Length: " << idisp << " - App. Force: " << aforce << std::endl;

		// Write specific outputs to file
		if (this_FE_process==0)
		{
			std::ofstream ofile;
			char fname[1024]; sprintf(fname, "%s/load_deflection.csv", macrologloc);

			if (timestep_no == 1){
				ofile.open (fname);
				if (ofile.is_open())
				{
					// writing the header of the file
					ofile << "timestep,time,gauge_length,applied_force" << std::endl;

					// writing the initial length of the gauge
					double ilength = 0.;
					double ytop = 0.;
					double ybot = 0.;
					for (typename DoFHandler<dim>::active_cell_iterator
							cell = dof_handler.begin_active();
							cell != dof_handler.end(); ++cell)
					{
						for(unsigned int ii=0;ii<lcga.size();ii++){
							if (cell->active_cell_index()==lcga[ii])
							{
								double ypos = cell->vertex(0)(1);
								if(ypos > 0.0) ytop = ypos;
								else ybot = ypos;
							}
						}
					}
					ilength = ytop-ybot;
					ofile << 0 << ", " << 0 << ", " << std::setprecision(16) << ilength << ", " << 0.0 << std::endl;
					ofile.close();
				}
				else std::cout << "Unable to open" << fname << " to write in it" << std::endl;
			}

			ofile.open (fname, std::ios::app);
			if (ofile.is_open())
			{
				ofile << timestep_no << ", " << present_time << ", " << std::setprecision(16) << idisp << ", " << aforce << std::endl;
				ofile.close();
			}
			else std::cout << "Unable to open" << fname << " to write in it" << std::endl;
		}

		// Cells of special interest (store atom dump of every update of each replica of each cell)
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
		{
			bool cell_is_of_special_interest = false;
			for (unsigned int i=0; i<lcis.size(); i++)
				if(cell->active_cell_index() == lcis[i]) cell_is_of_special_interest = true;

			if (cell_is_of_special_interest)
				if (cell->is_locally_owned())
				{
					PointHistory<dim> *local_quadrature_points_history
							= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

					char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());
					char filename[1024];

					// Save box state at all timesteps
					for(unsigned int repl=1;repl<nrepl+1;repl++)
					{
						sprintf(filename, "%s/last.%s.%s_%d.atom", nanostatelocout, cell_id,
								local_quadrature_points_history[0].mat.c_str(), repl);
						std::ifstream  nanoin(filename, std::ios::binary);
						// Also check if file has changed since last timestep
						if (nanoin.good()){
							sprintf(filename, "%s/%d.%s.%s_%d.atom", nanologlocsi, timestep_no, cell_id,
									local_quadrature_points_history[0].mat.c_str(), repl);
							std::ofstream  nanoout(filename,   std::ios::binary);
							nanoout << nanoin.rdbuf();
							nanoin.close();
							nanoout.close();
						}
					}
				}
			// Remove all dump atom files in the out folder
			if (cell->is_locally_owned()){
				PointHistory<dim> *local_quadrature_points_history
						= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());
				char filename[1024];

				for(unsigned int repl=1;repl<nrepl+1;repl++)
				{
					// Removing atom dump for all replicas of all cells (
					sprintf(filename, "%s/last.%s.%s_%d.atom", nanostatelocout, cell_id,
								local_quadrature_points_history[0].mat.c_str(), repl);
					remove(filename);
				}
			}
		}
	}



	template <int dim>
		void FEProblem<dim>::output_visualisation (const double present_time, const int timestep_no)
		{
			// Data structure for VTK output
			DataOut<dim> data_out;
			data_out.attach_dof_handler (dof_handler);

			// Output of displacement as a vector
			std::vector<DataComponentInterpretation::DataComponentInterpretation>
			data_component_interpretation
			(dim, DataComponentInterpretation::component_is_part_of_vector);
			std::vector<std::string>  displacement_names (dim, "displacement");
			data_out.add_data_vector (displacement,
					displacement_names,
					DataOut<dim>::type_dof_data,
					data_component_interpretation);

			// Output of error per cell as a scalar
			//data_out.add_data_vector (error_per_cell, "error_per_cell");

			// Output of the cell averaged striffness over quadrature
			// points as a scalar in direction 0000, 1111 and 2222
			std::vector<Vector<double> > avg_stiff (dim,
					Vector<double>(triangulation.n_active_cells()));
			for (int i=0;i<dim;++i){
				{
					typename Triangulation<dim>::active_cell_iterator
					cell = triangulation.begin_active(),
					endc = triangulation.end();
					for (; cell!=endc; ++cell)
						if (cell->is_locally_owned())
						{
							double accumulated_stiffi = 0.;
							for (unsigned int q=0;q<quadrature_formula.size();++q)
								accumulated_stiffi += reinterpret_cast<PointHistory<dim>*>
									(cell->user_pointer())[q].new_stiff[i][i][i][i];

							avg_stiff[i](cell->active_cell_index()) = accumulated_stiffi/quadrature_formula.size();
						}
						else avg_stiff[i](cell->active_cell_index()) = -1e+20;
				}
				std::string si = std::to_string(i);
				std::string name = "stiffness_"+si+si+si+si;
				data_out.add_data_vector (avg_stiff[i], name);
			}


			// Output of the cell norm of the averaged strain tensor over quadrature
			// points as a scalar
			Vector<double> norm_of_strain (triangulation.n_active_cells());
			{
				typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						SymmetricTensor<2,dim> accumulated_strain;
						for (unsigned int q=0;q<quadrature_formula.size();++q)
							accumulated_strain += reinterpret_cast<PointHistory<dim>*>
						(cell->user_pointer())[q].new_strain;

						norm_of_strain(cell->active_cell_index())
						= (accumulated_strain / quadrature_formula.size()).norm();
					}
					else norm_of_strain(cell->active_cell_index()) = -1e+20;
			}
			data_out.add_data_vector (norm_of_strain, "norm_of_strain");

			// Output of the cell norm of the averaged stress tensor over quadrature
			// points as a scalar
			Vector<double> norm_of_stress (triangulation.n_active_cells());
			{
				typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						SymmetricTensor<2,dim> accumulated_stress;
						for (unsigned int q=0;q<quadrature_formula.size();++q)
							accumulated_stress += reinterpret_cast<PointHistory<dim>*>
						(cell->user_pointer())[q].new_stress;

						norm_of_stress(cell->active_cell_index())
						= (accumulated_stress / quadrature_formula.size()).norm();
					}
					else norm_of_stress(cell->active_cell_index()) = -1e+20;
			}
			data_out.add_data_vector (norm_of_stress, "norm_of_stress");

			// Output of the cell XX-component of the averaged stress tensor over quadrature
			// points as a scalar
			Vector<double> xx_stress (triangulation.n_active_cells());
			{
				typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						SymmetricTensor<2,dim> accumulated_stress;
						for (unsigned int q=0;q<quadrature_formula.size();++q)
							accumulated_stress += reinterpret_cast<PointHistory<dim>*>
						(cell->user_pointer())[q].new_stress;

						xx_stress(cell->active_cell_index())
						= (accumulated_stress[0][0] / quadrature_formula.size());
					}
					else xx_stress(cell->active_cell_index()) = -1e+20;
			}
			data_out.add_data_vector (xx_stress, "stress_00");

			// Output of the cell YY-component of the averaged stress tensor over quadrature
			// points as a scalar
			Vector<double> yy_stress (triangulation.n_active_cells());
			{
				typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						SymmetricTensor<2,dim> accumulated_stress;
						for (unsigned int q=0;q<quadrature_formula.size();++q)
							accumulated_stress += reinterpret_cast<PointHistory<dim>*>
						(cell->user_pointer())[q].new_stress;

						yy_stress(cell->active_cell_index())
						= (accumulated_stress[1][1] / quadrature_formula.size());
					}
					else yy_stress(cell->active_cell_index()) = -1e+20;
			}
			data_out.add_data_vector (yy_stress, "stress_11");

			// Output of the partitioning of the mesh on processors
			std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());
			GridTools::get_subdomain_association (triangulation, partition_int);
			const Vector<double> partitioning(partition_int.begin(),
					partition_int.end());
			data_out.add_data_vector (partitioning, "partitioning");

			data_out.build_patches ();

			// Grouping spatially partitioned outputs
			std::string smacrologloc(macrologloc);
			std::string filename = smacrologloc + "/" + "solution-" + Utilities::int_to_string(timestep_no,4)
			+ "." + Utilities::int_to_string(this_FE_process,3)
			+ ".vtu";
			AssertThrow (n_FE_processes < 1000, ExcNotImplemented());

			std::ofstream output (filename.c_str());
			data_out.write_vtu (output);

			if (this_FE_process==0)
			{
				std::vector<std::string> filenames_loc;
				for (int i=0; i<n_FE_processes; ++i)
					filenames_loc.push_back ("solution-" + Utilities::int_to_string(timestep_no,4)
				+ "." + Utilities::int_to_string(i,3)
				+ ".vtu");

				const std::string
				visit_master_filename = (smacrologloc + "/" + "solution-" +
						Utilities::int_to_string(timestep_no,4) +
						".visit");
				std::ofstream visit_master (visit_master_filename.c_str());
				//data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
				DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

				const std::string
				pvtu_master_filename = (smacrologloc + "/" + "solution-" +
						Utilities::int_to_string(timestep_no,4) +
						".pvtu");
				std::ofstream pvtu_master (pvtu_master_filename.c_str());
				data_out.write_pvtu_record (pvtu_master, filenames_loc);

				static std::vector<std::pair<double,std::string> > times_and_names;
				const std::string
							pvtu_master_filename_loc = ("solution-" +
									Utilities::int_to_string(timestep_no,4) +
									".pvtu");
				times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename_loc));
				std::ofstream pvd_output (smacrologloc + "/" + "solution.pvd");
				//data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
				DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
			}
		}



	template <int dim>
	void FEProblem<dim>::output_results (const double present_time, const int timestep_no, unsigned int nrepl, char* nanostatelocout, char* nanologlocsi)
	{
		// Output local history by processor
		output_lhistory (present_time);

		// Specific outputs to the current test
		output_specific (present_time, timestep_no, nrepl, nanostatelocout, nanologlocsi);

		// Output visualisation files for paraview
		output_visualisation(present_time, timestep_no);
	}



	template <int dim>
	void FEProblem<dim>::restart_save (const double present_time, char* nanostatelocout, char* nanostatelocres, unsigned int nrepl) const
	{
		char filename[1024];

		// Copy of the solution vector at the end of the presently converged time-step.
		if (this_FE_process==0)
		{
			// Write solution vector to binary for simulation restart
			std::string smacrostatelocrestmp(macrostatelocres);
			const std::string solution_filename = (smacrostatelocrestmp + "/" + "lcts.solution.bin");
			std::ofstream ofile(solution_filename);
			displacement.block_write(ofile);
			ofile.close();
		}


		// Output of the last converged timestep local history per processor
		sprintf(filename, "%s/lcts.pr_%d.lhistory.bin", macrostatelocres, this_FE_process);
		std::ofstream  lhprocoutbin(filename, std::ios_base::binary);

		// Output of complete local history in a single file per processor
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());

				PointHistory<dim> *local_qp_hist
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

				// Save strain, updstrain, stress history in one file per proc
				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					lhprocoutbin << present_time
							<< "," << cell->active_cell_index()
							<< "," << q
							<< "," << local_qp_hist[q].mat.c_str();
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							lhprocoutbin << "," << std::setprecision(16) << local_qp_hist[q].upd_strain[k][l];
						}
					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							lhprocoutbin << "," << std::setprecision(16) << local_qp_hist[q].new_stress[k][l];
						}
					lhprocoutbin << std::endl;
				}
			}
		lhprocoutbin.close();

		// Copy of the last updated state of the MD boxes of each cell
		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());

				PointHistory<dim> *local_quadrature_points_history
					= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

				// Save box state history
				for(unsigned int repl=1;repl<nrepl+1;repl++)
				{
					sprintf(filename, "%s/last.%s.%s_%d.bin", nanostatelocout, cell_id,
							local_quadrature_points_history[0].mat.c_str(), repl);
					std::ifstream  nanoin(filename, std::ios::binary);
					if (nanoin.good()){
						sprintf(filename, "%s/lcts.%s.%s_%d.bin", nanostatelocres, cell_id,
								local_quadrature_points_history[0].mat.c_str(), repl);
						std::ofstream  nanoout(filename,   std::ios::binary);
						nanoout << nanoin.rdbuf();
						nanoin.close();
						nanoout.close();
					}
				}
			}
		MPI_Barrier(FE_communicator);
	}



	template <int dim>
	void FEProblem<dim>::restart_system (char* nanostatelocin, char* nanostatelocout, unsigned int nrepl)
	{
		char filename[1024];

		// Recovery of the solution vector containing total displacements in the
		// previous simulation and computing the total strain from it.
		sprintf(filename, "%s/restart/lcts.solution.bin", macrostatelocin);
		std::ifstream ifile(filename);
		if (ifile.is_open())
		{
			dcout << "    ...recovery of the position vector... " << std::flush;
			displacement.block_read(ifile);
			dcout << "    solution norm: " << displacement.l2_norm() << std::endl;
			ifile.close();

			dcout << "    ...computation of total strains from the recovered position vector. " << std::endl;
			FEValues<dim> fe_values (fe, quadrature_formula,
					update_values | update_gradients);
			std::vector<std::vector<Tensor<1,dim> > >
			solution_grads (quadrature_formula.size(),
					std::vector<Tensor<1,dim> >(dim));

			for (typename DoFHandler<dim>::active_cell_iterator
					cell = dof_handler.begin_active();
					cell != dof_handler.end(); ++cell)
				if (cell->is_locally_owned())
				{
					PointHistory<dim> *local_quadrature_points_history
					= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
					Assert (local_quadrature_points_history >=
							&quadrature_point_history.front(),
							ExcInternalError());
					Assert (local_quadrature_points_history <
							&quadrature_point_history.back(),
							ExcInternalError());
					fe_values.reinit (cell);
					fe_values.get_function_gradients (displacement,
							solution_grads);

					for (unsigned int q=0; q<quadrature_formula.size(); ++q)
					{
						// Strain tensor update
						local_quadrature_points_history[q].new_strain =
								get_strain (solution_grads[q]);

						// Only needed if the mesh is modified after every timestep...
						/*const Tensor<2,dim> rotation
						= get_rotation_matrix (solution_grads[q]);

						const SymmetricTensor<2,dim> rotated_new_strain
						= symmetrize(transpose(rotation) *
								static_cast<Tensor<2,dim> >
						(local_quadrature_points_history[q].new_strain) *
						rotation);

						local_quadrature_points_history[q].new_strain
						= rotated_new_strain;*/
					}
				}
		}

		// Opening processor local history file
		sprintf(filename, "%s/restart/lcts.pr_%d.lhistory.bin", macrostatelocin, this_FE_process);
		std::ifstream  lhprocin(filename, std::ios_base::binary);

		// If openend, restore local data history...
		int ncell_lhistory=0;
		if (lhprocin.good()){
			std::string line;
			// Compute number of cells in local history ()
			while(getline(lhprocin, line)){
				//nline_lhistory++;
				// Extract values...
				std::istringstream sline(line);
				std::string var;
				int item_count = 0;
				int cell = 0;
				while(getline(sline, var, ',' )){
					if(item_count==1) cell = std::stoi(var);
					item_count++;
				}
				ncell_lhistory = std::max(ncell_lhistory, cell);
			}
			//int ncell_lhistory = n_FE_processes*nline_lhistory/quadrature_formula.size();
			//std::cout << "proc: " << this_FE_process << " ncell history: " << ncell_lhistory << std::endl;

			// Create structure to store retrieve data as matrix[cell][qpoint]
			std::vector<std::vector<PointHistory<dim>> > proc_lhistory (ncell_lhistory+1,
					std::vector<PointHistory<dim> >(quadrature_formula.size()));

			MPI_Barrier(FE_communicator);

			// Read and insert data
			lhprocin.clear();
			lhprocin.seekg(0, std::ios_base::beg);
			while(getline(lhprocin, line)){
				// Extract values...
				std::istringstream sline(line);
				std::string var;
				int item_count = 0;
				int cell = 0;
				int qpoint = 0;
				while(getline(sline, var, ',' )){
					if(item_count==1) cell = std::stoi(var);
					else if(item_count==2) qpoint = std::stoi(var);
					else if(item_count==4) proc_lhistory[cell][qpoint].upd_strain[0][0] = std::stod(var);
					else if(item_count==5) proc_lhistory[cell][qpoint].upd_strain[0][1] = std::stod(var);
					else if(item_count==6) proc_lhistory[cell][qpoint].upd_strain[0][2] = std::stod(var);
					else if(item_count==7) proc_lhistory[cell][qpoint].upd_strain[1][1] = std::stod(var);
					else if(item_count==8) proc_lhistory[cell][qpoint].upd_strain[1][2] = std::stod(var);
					else if(item_count==9) proc_lhistory[cell][qpoint].upd_strain[2][2] = std::stod(var);
					else if(item_count==10) proc_lhistory[cell][qpoint].new_stress[0][0] = std::stod(var);
					else if(item_count==11) proc_lhistory[cell][qpoint].new_stress[0][1] = std::stod(var);
					else if(item_count==12) proc_lhistory[cell][qpoint].new_stress[0][2] = std::stod(var);
					else if(item_count==13) proc_lhistory[cell][qpoint].new_stress[1][1] = std::stod(var);
					else if(item_count==14) proc_lhistory[cell][qpoint].new_stress[1][2] = std::stod(var);
					else if(item_count==15) proc_lhistory[cell][qpoint].new_stress[2][2] = std::stod(var);
					item_count++;
				}
//				if(cell%90 == 0) std::cout << cell<<","<<qpoint<<","<<proc_lhistory[cell][qpoint].upd_strain[0][0]
//				    <<","<<proc_lhistory[cell][qpoint].new_stress[0][0] << std::endl;
			}

			MPI_Barrier(FE_communicator);

			// Need to verify that the recovery of the local history is performed correctly...
			dcout << "    ...recovery of the quadrature point history. " << std::endl;
			for (typename DoFHandler<dim>::active_cell_iterator
					cell = dof_handler.begin_active();
					cell != dof_handler.end(); ++cell)
				if (cell->is_locally_owned())
				{
					PointHistory<dim> *local_quadrature_points_history
					= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
					Assert (local_quadrature_points_history >=
							&quadrature_point_history.front(),
							ExcInternalError());
					Assert (local_quadrature_points_history <
							&quadrature_point_history.back(),
							ExcInternalError());

					for (unsigned int q=0; q<quadrature_formula.size(); ++q)
					{
						//std::cout << "proc: " << this_FE_process << " cell: " << cell->active_cell_index() << " qpoint: " << q << std::endl;
						// Assigning update strain and stress tensor
						local_quadrature_points_history[q].upd_strain=proc_lhistory[cell->active_cell_index()][q].upd_strain;
						local_quadrature_points_history[q].new_stress=proc_lhistory[cell->active_cell_index()][q].new_stress;

						// Restore box state history
						for(unsigned int repl=1;repl<nrepl+1;repl++)
						{
							sprintf(filename, "%s/restart/lcts.%d.%s_%d.bin", nanostatelocin, cell->active_cell_index(),
									local_quadrature_points_history[0].mat.c_str(), repl);
							std::ifstream  nanoin(filename, std::ios::binary);
							if (nanoin.good()){
								sprintf(filename, "%s/last.%d.%s_%d.bin", nanostatelocout, cell->active_cell_index(),
										local_quadrature_points_history[0].mat.c_str(), repl);
								std::ofstream  nanoout(filename,   std::ios::binary);
								nanoout << nanoin.rdbuf();
								nanoin.close();
								nanoout.close();
							}
						}
					}
				}
		}
	}



	template <int dim>
	void FEProblem<dim>::clean_transfer(unsigned int nrepl)
	{
		char filename[1024];

		// Removing lists of material types of quadrature points to update per procs
		sprintf(filename, "%s/last.%d.matqpupdates", macrostatelocout, this_FE_process);
		remove(filename);

		// Removing lists of quadrature points to update per procs
		sprintf(filename, "%s/last.%d.qpupdates", macrostatelocout, this_FE_process);
		remove(filename);

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());

				for(unsigned int repl=1;repl<nrepl+1;repl++)
				{
					// Removing replica stiffness passing file used to average cell stiffness
					//sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout, cell_id, repl);
					//remove(filename);

					// Removing replica stress passing file used to average cell stress
					sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout, cell_id, repl);
					remove(filename);
				}

				// Removing stiffness passing file
				//sprintf(filename, "%s/last.%s.stiff", macrostatelocout, cell_id);
				//remove(filename);

				// Removing stress passing file
				sprintf(filename, "%s/last.%s.stress", macrostatelocout, cell_id);
				remove(filename);

				// Removing updstrain passing file
				sprintf(filename, "%s/last.%s.upstrain", macrostatelocout, cell_id);
				remove(filename);
			}
	}





	template <int dim>
	class HMMProblem
	{
	public:
		HMMProblem ();
		~HMMProblem ();
		void run ();

	private:
		void clean_temporary (FEProblem<dim> &fe_problem);
		void set_repositories ();
		void set_dealii_procs (int npd);
		// void init_lammps_procs ();
		void set_lammps_procs (int nmdruns);
		void initialize_replicas ();
		void do_timestep (FEProblem<dim> &fe_problem);
		void solve_timestep (FEProblem<dim> &fe_problem);

		void run_single_md(char* ctime, char* ccell, const char* cmat, unsigned int repl);
		void update_cells_with_molecular_dynamics ();

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		MPI_Comm 							dealii_communicator;
		int									root_dealii_process;
		int 								n_dealii_processes;
		int 								this_dealii_process;
		int 								dealii_pcolor;

		// MPI_Comm 							lammps_global_communicator;
		MPI_Comm 							lammps_batch_communicator;
		// int 								n_lammps_processes;
		int 								n_lammps_processes_per_batch;
		int 								n_lammps_batch;
		int 								this_lammps_process;
		int 								this_lammps_batch_process;
		int 								lammps_pcolor;
		int									machine_ppn;

		ConditionalOStream 					hcout;

		double              				present_time;
		double              				present_timestep;
		double              				end_time;
		int        							timestep_no;
		int        							newtonstep_no;
		bool 								updated_md;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;

		char                                macrostateloc[1024];
		char                                macrostatelocin[1024];
		char                                macrostatelocout[1024];
		char                                macrostatelocres[1024];
		char                                macrologloc[1024];

		char                                nanostateloc[1024];
		char                                nanostatelocin[1024];
		char                                nanostatelocout[1024];
		char                                nanostatelocres[1024];
		char                                nanologloc[1024];
		char                                nanologlocsi[1024];

	};



	template <int dim>
	HMMProblem<dim>::HMMProblem ()
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	HMMProblem<dim>::~HMMProblem ()
	{}



	template <int dim>
	void HMMProblem<dim>::update_cells_with_molecular_dynamics()
	{
		//char prev_time_id[1024]; sprintf(prev_time_id, "%d-%d", timestep_no, newtonstep_no-1);
		char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);

		// Check list of files corresponding to current "time_id"
		int ncupd = 0;
		char filenamelist[1024];
		sprintf(filenamelist, "%s/last.qpupdates", macrostatelocout);
		std::ifstream ifile;
		std::string iline;

		// Count number of cells to update
		ifile.open (filenamelist);
		if (ifile.is_open())
		{
			while (getline(ifile, iline)) ncupd++;
			ifile.close();
		}
		else hcout << "Unable to open" << filenamelist << " to read it" << std::endl;

		// Flag to know if some local cells stiffness are updated
		if (ncupd>0) updated_md = true;

		hcout << "        " << "...are some stiffnesses updated in that call to update_stiffness_with_molecular_dynamics? " << updated_md << std::endl;

		if (ncupd>0){
			// Create list of quadid
			char **cell_id = new char *[ncupd];
			for (int c=0; c<ncupd; ++c) cell_id[c] = new char[1024];

			ifile.open (filenamelist);
			int nline = 0;
			while (nline<ncupd && ifile.getline(cell_id[nline], sizeof(cell_id[nline]))) nline++;
			ifile.close();

			// Load material type of cells to be updated
			std::vector<std::string> matcellupd (ncupd);
			sprintf(filenamelist, "%s/last.matqpupdates", macrostatelocout);
			ifile.open (filenamelist);
			nline = 0;
			while (nline<ncupd && std::getline(ifile, matcellupd[nline])) nline++;
			ifile.close();

			// Number of MD simulations at this iteration...
			int nmdruns = ncupd*nrepl;

			// Setting up batch of processes
			set_lammps_procs(nmdruns);

			MPI_Barrier(world_communicator);

			// Computing cell state update running one simulation per MD replica (basic job scheduling and executing)
			hcout << "        " << "...dispatching the MD runs on batch of processes..." << std::endl;
			hcout << "        " << "...cells and replicas completed: " << std::flush;
			for (int c=0; c<ncupd; ++c)
			{
				for(unsigned int repl=1;repl<nrepl+1;repl++)
				{
					// The variable 'imdrun' assigned to a run is a multiple of the batch number the run will be run on
					int imdrun=c*nrepl + (repl-1);

					// Allocation of a MD run to a batch of processes
					if (lammps_pcolor == (imdrun%n_lammps_batch)) run_single_md(time_id, cell_id[c], matcellupd[c].c_str(), repl);
				}
			}
			hcout << std::endl;

			MPI_Barrier(world_communicator);

			// Verify the integrity of the stiffness tensor (constraint C_upd>stol*C_ini)
			/*double stol = 0.001;

			for (int c=0; c<ncupd; ++c)
			{
				if (lammps_pcolor == (c%n_lammps_batch))
				{
					if(this_lammps_batch_process == 0)
					{
						SymmetricTensor<4,dim> loc_stiffness;
						char filename[1024];

						for(unsigned int repl=1;repl<nrepl+1;repl++)
						{
							SymmetricTensor<4,dim> loc_upd_rep_stiffness;
							sprintf(filename, "%s/last.%s.%s_%d.stiff", macrostatelocout, cell_id[c], matcellupd[c].c_str(), repl);
							read_tensor<dim>(filename, loc_upd_rep_stiffness);

							SymmetricTensor<4,dim> loc_ini_rep_stiffness;
							sprintf(filename, "%s/init.%s_%d.stiff", macrostatelocout, matcellupd[c].c_str(), repl);
							read_tensor<dim>(filename, loc_ini_rep_stiffness);

							for(unsigned int k=0;k<dim;k++)
								for(unsigned int l=k;l<dim;l++)
									for(unsigned int m=0;m<dim;m++)
										for(unsigned int n=m;n<dim;n++)
											if(fabs(loc_upd_rep_stiffness[k][l][m][n]) < stol*loc_ini_rep_stiffness[k][l][m][n])
											{
												//std::cout << "               "
												//		  << "Cell: " << cell_id[c] << " Replica: " << repl
												//		  << " required stiffness correction !!"
												//		  << std::endl;
												loc_upd_rep_stiffness[k][l][m][n] *= stol*
														loc_ini_rep_stiffness[k][l][m][n]/fabs(loc_upd_rep_stiffness[k][l][m][n]);
											}

							sprintf(filename, "%s/last.%s.%s_%d.stiff", macrostatelocout, cell_id[c], matcellupd[c].c_str(), repl);
							write_tensor<dim>(filename, loc_upd_rep_stiffness);
						}
					}
				}
			}

			MPI_Barrier(world_communicator);*/

			// Averaging stiffness and stress per cell over replicas
			for (int c=0; c<ncupd; ++c)
			{
				if (lammps_pcolor == (c%n_lammps_batch))
				{
					// Write the new stress and stiffness tensors into two files, respectively
					// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
					if(this_lammps_batch_process == 0)
					{
						//SymmetricTensor<4,dim> loc_stiffness;
						SymmetricTensor<2,dim> loc_stress;
						char filename[1024];

						for(unsigned int repl=1;repl<nrepl+1;repl++)
						{
							/*SymmetricTensor<4,dim> loc_rep_stiffness;
							sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout, cell_id[c], repl);
							read_tensor<dim>(filename, loc_rep_stiffness);

							loc_stiffness += loc_rep_stiffness;*/

							SymmetricTensor<2,dim> loc_rep_stress;
							sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout, cell_id[c], repl);
							read_tensor<dim>(filename, loc_rep_stress);

							loc_stress += loc_rep_stress;
						}

						//loc_stiffness /= nrepl;
						loc_stress /= nrepl;

						/*sprintf(filename, "%s/last.%s.stiff", macrostatelocout, cell_id[c]);
						write_tensor<dim>(filename, loc_stiffness);*/

						sprintf(filename, "%s/last.%s.stress", macrostatelocout, cell_id[c]);
						write_tensor<dim>(filename, loc_stress);

					}
				}
			}
		}
	}



	template <int dim>
	void HMMProblem<dim>::solve_timestep (FEProblem<dim> &fe_problem)
	{
		double previous_res;

		do
		{
			hcout << "  Initial assembling FE system..." << std::flush;
			if(dealii_pcolor==0) previous_res = fe_problem.assemble_system ();
			hcout << "  Initial residual: "
					<< previous_res
					<< std::endl;

			updated_md = false;

			for (unsigned int inner_iteration=0; inner_iteration<2; ++inner_iteration)
			{
				++newtonstep_no;
				hcout << "    Beginning of timestep: " << timestep_no << " - newton step: " << newtonstep_no << std::flush;
				hcout << "    Solving FE system..." << std::flush;
				if(dealii_pcolor==0) fe_problem.solve_linear_problem_CG();

				MPI_Barrier(world_communicator);
				hcout << "    Updating quadrature point data..." << std::endl;

				if(dealii_pcolor==0) fe_problem.update_strain_quadrature_point_history
						(fe_problem.newton_update_displacement, timestep_no, newtonstep_no, updated_md);
				MPI_Barrier(world_communicator);

				hcout << "    Have some stiffnesses been updated in this group of iterations? " << updated_md << std::endl;

				if (!updated_md) update_cells_with_molecular_dynamics();
				MPI_Barrier(world_communicator);

				if(dealii_pcolor==0) fe_problem.update_stress_quadrature_point_history
						(fe_problem.newton_update_displacement, timestep_no, newtonstep_no);

				hcout << "    Re-assembling FE system..." << std::flush;
				if(dealii_pcolor==0) previous_res = fe_problem.assemble_system ();
				MPI_Barrier(world_communicator);

				// Cleaning temporary files (nanoscale logs and FE/MD data transfer)
				clean_temporary(fe_problem);

				MPI_Barrier(world_communicator);

				// Share the value of previous_res in between processors
				MPI_Bcast(&previous_res, 1, MPI_DOUBLE, root_dealii_process, world_communicator);

				hcout << "    Residual: "
						<< previous_res
						<< std::endl;
			}
		} while (previous_res>1e-02 || updated_md);
	}




	template <int dim>
	void HMMProblem<dim>::do_timestep (FEProblem<dim> &fe_problem)
	{
		// Frequencies of output and save
		int freq_restart_output = 1;
		int freq_output_results = 1;

		// Updating time variable
		present_time += present_timestep;
		++timestep_no;
		hcout << "Timestep " << timestep_no << " at time " << present_time
				<< std::endl;
		if (present_time > end_time)
		{
			present_timestep -= (present_time - end_time);
			present_time = end_time;
		}

		// Initialisation of timestep variables
		newtonstep_no = 0;
		updated_md = false;
		if(dealii_pcolor==0) fe_problem.incremental_displacement = 0;

		// Setting boudary conditions for current timestep
		if(dealii_pcolor==0) fe_problem.set_boundary_values ();

		// Updating current strains and stresses with the boundary conditions information
		if(dealii_pcolor==0) fe_problem.update_strain_quadrature_point_history (fe_problem.incremental_displacement, timestep_no, newtonstep_no, updated_md);
		if(dealii_pcolor==0) fe_problem.update_stress_quadrature_point_history (fe_problem.incremental_displacement, timestep_no, newtonstep_no);
		MPI_Barrier(world_communicator);

		// Solving iteratively the current timestep
		solve_timestep (fe_problem);

		// Updating the total displacement and velocity vectors
		if(dealii_pcolor==0)fe_problem.displacement+=fe_problem.incremental_displacement;

		//if(dealii_pcolor==0) fe_problem.error_estimation ();

		// Outputs
		if(dealii_pcolor==0) if(timestep_no%freq_output_results==0)  fe_problem.output_results (present_time, timestep_no, nrepl, nanostatelocout, nanologlocsi);

		// Saving files for restart
		if(dealii_pcolor==0) if(timestep_no%freq_restart_output==0) fe_problem.restart_save (present_time, nanostatelocout, nanostatelocres, nrepl);

		MPI_Barrier(world_communicator);

		hcout << std::endl;
	}




	template <int dim>
	void HMMProblem<dim>::initialize_replicas ()
	{
		// Number of MD simulations at this iteration...
		int nmdruns = mdtype.size()*nrepl;

		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		set_lammps_procs(nmdruns);

		for(unsigned int imdt=0;imdt<mdtype.size();imdt++)
		{
			// type of MD box (so far PE or PNC)
			std::string mdt = mdtype[imdt];

			for(unsigned int repl=1;repl<nrepl+1;repl++)
			{
				int imdrun=imdt*nrepl + (repl-1);
				if (lammps_pcolor == (imdrun%n_lammps_batch))
				{
					std::vector<double> 				initial_length (dim);
					SymmetricTensor<2,dim> 				initial_stress_tensor;
					SymmetricTensor<4,dim> 				initial_stiffness_tensor;

					char macrofilenamein[1024];
					sprintf(macrofilenamein, "%s/init.%s_%d.stiff", macrostatelocin, mdt.c_str(), repl);
					char macrofilenameout[1024];
					sprintf(macrofilenameout, "%s/init.%s_%d.stiff", macrostatelocout, mdt.c_str(), repl);
					bool macrostate_exists = file_exists(macrofilenamein);

					char macrofilenameinstress[1024];
					sprintf(macrofilenameinstress, "%s/init.%s_%d.stress", macrostatelocin, mdt.c_str(), repl);
					char macrofilenameoutstress[1024];
					sprintf(macrofilenameoutstress, "%s/init.%s_%d.stress", macrostatelocout, mdt.c_str(), repl);
					bool macrostatestress_exists = file_exists(macrofilenameinstress);

					char macrofilenameinlength[1024];
					sprintf(macrofilenameinlength, "%s/init.%s_%d.length", macrostatelocin, mdt.c_str(), repl);
					char macrofilenameoutlength[1024];
					sprintf(macrofilenameoutlength, "%s/init.%s_%d.length", macrostatelocout, mdt.c_str(), repl);
					bool macrostatelength_exists = file_exists(macrofilenameinlength);

					char nanofilenamein[1024];
					sprintf(nanofilenamein, "%s/init.%s_%d.bin", nanostatelocin, mdt.c_str(), repl);
					char nanofilenameout[1024];
					sprintf(nanofilenameout, "%s/init.%s_%d.bin", nanostatelocout, mdt.c_str(), repl);
					bool nanostate_exists = file_exists(nanofilenamein);

					if(!macrostate_exists || !macrostatestress_exists || !macrostatelength_exists || !nanostate_exists){
						if(this_lammps_batch_process == 0) std::cout << "        (type " << mdt << " - repl "<< repl << ") ...from a molecular dynamics simulation       " << std::endl;
						lammps_initiation<dim> (initial_stress_tensor, initial_stiffness_tensor, initial_length, lammps_batch_communicator,
								nanostatelocin, nanostatelocout, nanologloc, mdt, repl);

						if(this_lammps_batch_process == 0) write_tensor<dim>(macrofilenameout, initial_stiffness_tensor);
						if(this_lammps_batch_process == 0) write_tensor<dim>(macrofilenameoutstress, initial_stress_tensor);
						if(this_lammps_batch_process == 0) write_tensor<dim>(macrofilenameoutlength, initial_length);
					}
					else{
						if(this_lammps_batch_process == 0){
							std::cout << " (repl "<< repl << ")  ...from an existing stiffness tensor       " << std::endl;
							std::ifstream  macroin(macrofilenamein, std::ios::binary);
							std::ofstream  macroout(macrofilenameout,   std::ios::binary);
							macroout << macroin.rdbuf();
							macroin.close();
							macroout.close();

							std::ifstream  macrostressin(macrofilenameinstress, std::ios::binary);
							std::ofstream  macrostressout(macrofilenameoutstress,   std::ios::binary);
							macrostressout << macrostressin.rdbuf();
							macrostressin.close();
							macrostressout.close();

							std::ifstream  macrolengthin(macrofilenameinlength, std::ios::binary);
							std::ofstream  macrolengthout(macrofilenameoutlength,   std::ios::binary);
							macrolengthout << macrolengthin.rdbuf();
							macrolengthin.close();
							macrolengthout.close();

							std::ifstream  nanoin(nanofilenamein, std::ios::binary);
							std::ofstream  nanoout(nanofilenameout,   std::ios::binary);
							nanoout << nanoin.rdbuf();
							nanoin.close();
							nanoout.close();
						}
					}
				}
			}
		}

		MPI_Barrier(world_communicator);

		if(this_lammps_batch_process == 0){
			for(unsigned int imd=0;imd<mdtype.size();imd++)
			{
				SymmetricTensor<4,dim> 				initial_ensemble_stiffness_tensor;
				initial_ensemble_stiffness_tensor = 0.;

				// type of MD box (so far PE or PNC)
				std::string mdt = mdtype[imd];

				for(unsigned int repl=1;repl<nrepl+1;repl++)
				{
					char macrofilenamein[1024];
					sprintf(macrofilenamein, "%s/init.%s_%d.stiff", macrostatelocout, mdt.c_str(), repl);

					SymmetricTensor<4,dim> 				initial_stiffness_tensor;
					read_tensor<dim>(macrofilenamein, initial_stiffness_tensor);

					initial_ensemble_stiffness_tensor += initial_stiffness_tensor;

				}

				initial_ensemble_stiffness_tensor /= nrepl;

				// Cleaning the stiffness tensor to remove negative diagonal terms and shear coupling terms...
				for(unsigned int k=0;k<dim;k++)
					for(unsigned int l=k;l<dim;l++)
						for(unsigned int m=0;m<dim;m++)
							for(unsigned int n=m;n<dim;n++)
								if(!((k==l && m==n) || (k==m && l==n))){
									//std::cout << "       ... removal of shear coupling terms" << std::endl;
									initial_ensemble_stiffness_tensor[k][l][m][n] *= 1.0;
								}
								// Does not make any sense for tangent stiffness...
								//else if(initial_ensemble_stiffness_tensor[k][l][m][n]<0.0) initial_ensemble_stiffness_tensor[k][l][m][n] *= +1.0; // correction -> *= -1.0

				char macrofilenameout[1024];
				sprintf(macrofilenameout, "%s/init.%s.stiff", macrostatelocout, mdt.c_str());

				write_tensor<dim>(macrofilenameout, initial_ensemble_stiffness_tensor);
			}
		}
	}




	// There are several number of processes encountered: (i) n_lammps_processes the highest provided
	// as an argument to aprun, (ii) ND the number of processes provided to deal.ii
	// [arbitrary], (iii) NI the number of processes provided to the lammps initiation
	// [as close as possible to n_world_processes], and (iv) n_lammps_processes_per_batch the number of processes provided to one lammps
	// testing [NT divided by n_lammps_batch the number of concurrent testing boxes].
	template <int dim>
	void HMMProblem<dim>::set_lammps_procs (int nmdruns)
	{
		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		int npbtch_min = machine_ppn;
		//int nbtch_max = int(n_world_processes/npbtch_min);

		//int nrounds = int(nmdruns/nbtch_max)+1;
		//int nmdruns_round = nmdruns/nrounds;

		int fair_npbtch = int(n_world_processes/(nmdruns));

		int npb = std::max(npbtch_min, fair_npbtch - fair_npbtch%npbtch_min);
		//int nbtch = int(n_world_processes/npbtch);

		// Arbitrary setting of NB and NT
		n_lammps_processes_per_batch = npb;

		n_lammps_batch = int(n_world_processes/n_lammps_processes_per_batch);
		if(n_lammps_batch == 0) {n_lammps_batch=1; n_lammps_processes_per_batch=n_world_processes;}

		hcout << "        " << "...number of processes per batches: " << n_lammps_processes_per_batch
							<< "   ...number of batches: " << n_lammps_batch << std::endl;

		lammps_pcolor = MPI_UNDEFINED;

		// LAMMPS processes color: regroup processes by batches of size NB, except
		// the last ones (me >= NB*NC) to create batches of only NB processes, nor smaller.
		if(this_world_process < n_lammps_processes_per_batch*n_lammps_batch)
			lammps_pcolor = int(this_world_process/n_lammps_processes_per_batch);
		// Initially we used MPI_UNDEFINED, but why waste processes... The processes over
		// 'n_lammps_processes_per_batch*n_lammps_batch' are assigned to the last batch...
		// finally it is better to waste them than failing the simulation with an odd number
		// of processes for the last batch
		/*else
			lammps_pcolor = int((n_lammps_processes_per_batch*n_lammps_batch-1)/n_lammps_processes_per_batch);
		*/

		// Definition of the communicators
		MPI_Comm_split(world_communicator, lammps_pcolor, this_world_process, &lammps_batch_communicator);
		MPI_Comm_rank(lammps_batch_communicator,&this_lammps_batch_process);

		// Recapitulating allocation of each process to deal and lammps
		/*std::cout << "        proc world rank: " << this_world_process
				<< " - deal color: " << dealii_pcolor
				<< " - lammps color: " << lammps_pcolor << std::endl;*/
	}




	template <int dim>
	void HMMProblem<dim>::set_dealii_procs (int npd)
	{
		root_dealii_process = 0;
		n_dealii_processes = npd;

		dealii_pcolor = MPI_UNDEFINED;

		// Color set above 0 for processors that are going to be used
		if (this_world_process >= root_dealii_process &&
				this_world_process < root_dealii_process + n_dealii_processes) dealii_pcolor = 0;
		else dealii_pcolor = 1;

		MPI_Comm_split(world_communicator, dealii_pcolor, this_world_process, &dealii_communicator);
		MPI_Comm_rank(dealii_communicator, &this_dealii_process);
	}




	template <int dim>
	void HMMProblem<dim>::run_single_md (char* ctime, char* ccell, const char* cmat, unsigned int repl)
	{
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
				lammps_batch_communicator,
				nanostatelocout,
				nanologloc,
				cmat,
				repl);

		if(this_lammps_batch_process == 0)
		{
			std::cout << " \t" << ccell <<"-"<< repl << " \t" << std::flush;

			/*sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout, ccell, repl);
			write_tensor<dim>(filename, loc_rep_stiffness);*/

			sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout, ccell, repl);
			write_tensor<dim>(filename, loc_rep_stress);
		}
	}




	template <int dim>
	void HMMProblem<dim>::set_repositories ()
	{
		sprintf(macrostateloc, "./macroscale_state"); mkdir(macrostateloc, ACCESSPERMS);
		sprintf(macrostatelocin, "%s/in", macrostateloc); mkdir(macrostatelocin, ACCESSPERMS);
		sprintf(macrostatelocout, "%s/out", macrostateloc); mkdir(macrostatelocout, ACCESSPERMS);
		sprintf(macrostatelocres, "%s/restart", macrostateloc); mkdir(macrostatelocres, ACCESSPERMS);
		sprintf(macrologloc, "./macroscale_log"); mkdir(macrologloc, ACCESSPERMS);

		sprintf(nanostateloc, "./nanoscale_state"); mkdir(nanostateloc, ACCESSPERMS);
		sprintf(nanostatelocin, "%s/in", nanostateloc); mkdir(nanostatelocin, ACCESSPERMS);
		sprintf(nanostatelocout, "%s/out", nanostateloc); mkdir(nanostatelocout, ACCESSPERMS);
		sprintf(nanostatelocres, "%s/restart", nanostateloc); mkdir(nanostatelocres, ACCESSPERMS);
		sprintf(nanologloc, "./nanoscale_log"); mkdir(nanologloc, ACCESSPERMS);
		sprintf(nanologlocsi, "%s/spec", nanologloc); mkdir(nanologlocsi, ACCESSPERMS);
	}




	template <int dim>
	void HMMProblem<dim>::clean_temporary (FEProblem<dim> &fe_problem)
	{
		if(dealii_pcolor==0) fe_problem.clean_transfer (nrepl);

		// Cleaning the log files for all the MD simulations of the current timestep
		if (this_world_process==0)
		{
			char command[1024];
			// Clean "nanoscale_logs" of the finished timestep
			for(unsigned int repl=1;repl<nrepl+1;repl++)
			{
				sprintf(command, "rm -rf %s/R%d/*", nanologloc, repl);
				system(command);
				//sprintf(command, "%s/R%d/*", nanologloc, repl);
				//boost::filesystem::remove_all(command);
			}
		}
	}




	template <int dim>
	void HMMProblem<dim>::run ()
	{
		// Current machine number of processes per node
		machine_ppn=16;

		// Set the dealii communicator using a limited amount of available processors
		// because dealii fails if processors do not have assigned cells. Plus, dealii
		// might not scale indefinitely
		set_dealii_procs(machine_ppn*20);

		// Setting repositories for input and creating repositories for outputs
		set_repositories();
		MPI_Barrier(world_communicator);

		hcout << "Building the HMM problem:       " << std::endl;

		// List of name of MD box types
		mdtype.push_back("g0");
		mdtype.push_back("g1");
		mdtype.push_back("g2");

		// Number of replicas in MD-ensemble
		nrepl=1;

		// Since LAMMPS is highly scalable, the initiation number of processes NI
		// can basically be equal to the maximum number of available processes NT which
		// can directly be found in the MPI_COMM.
		hcout << " Initialization of the Molecular Dynamics replicas...       " << std::endl;
		initialize_replicas();
		MPI_Barrier(world_communicator);

		// Initialization of time variables
		present_time = 0;
		present_timestep = 1;
		end_time = 200;
		timestep_no = 0;

		// Initiatilization of the FE problem
		hcout << " Initiation of the Finite Element problem...       " << std::endl;
		FEProblem<dim> fe_problem (dealii_communicator, dealii_pcolor,
				                    macrostatelocin, macrostatelocout, macrostatelocres, macrologloc,
									mdtype);
		MPI_Barrier(world_communicator);

		hcout << " Initiation of the Mesh...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.make_grid ();

		hcout << " Initiation of the global vectors and tensor...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.setup_system ();

		hcout << " Initiation of the local tensors...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.setup_quadrature_point_history ();

		hcout << " Loading previous simulation data...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.restart_system (nanostatelocin, nanostatelocout, nrepl);

		MPI_Barrier(world_communicator);

		// Running the solution algorithm of the FE problem
		hcout << "Beginning of incremental solution algorithm:       " << std::endl;
		while (present_time < end_time)
			do_timestep (fe_problem);

	}
}



int main (int argc, char **argv)
{
	try
	{
		using namespace HMM;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		HMMProblem<3> hmm_problem;
		hmm_problem.run();
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
