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
		SymmetricTensor<2,dim> old_stress;
		SymmetricTensor<2,dim> new_stress;
		SymmetricTensor<4,dim> old_stiff;
		SymmetricTensor<4,dim> new_stiff;
		SymmetricTensor<2,dim> old_strain;
		SymmetricTensor<2,dim> new_strain;
		SymmetricTensor<2,dim> inc_strain;
		SymmetricTensor<2,dim> upd_strain;
		bool to_be_updated;
	};

	bool file_exists(const char* file) {
		struct stat buf;
		return (stat(file, &buf) == 0);
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
	lammps_state (void *lmp, char *location, SymmetricTensor<2,dim>& stresses, SymmetricTensor<4,dim>& stiffnesses)
	{
		int me;
		MPI_Comm_rank(MPI_COMM_WORLD, &me);

		SymmetricTensor<2,2*dim> tmp;

		char cfile[1024];
		char cline[1024];

		sprintf(cline, "variable locbe string %s/%s", location, "ELASTIC");
		lammps_command(lmp,cline);

		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal 100"); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal 100"); lammps_command(lmp,cline);
		sprintf(cline, "variable nsstrain  equal 100"); lammps_command(lmp,cline);

		// Set strain perturbation amplitude
		sprintf(cline, "variable up equal 5.0e-3"); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress and the
		// stiffness tensors
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
				stresses[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.01325e+05;
			}

		// Filling the 6x6 Voigt Sitffness tensor with its computed as variables
		// by LAMMPS and conversion from GPa to Pa
		for(unsigned int k=0;k<2*dim;k++)
			for(unsigned int l=k;l<2*dim;l++)
			{
				char vcoef[1024];
				sprintf(vcoef, "C%d%dall", k+1, l+1);
				tmp[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;

				// In case problmes arise due to negative terms of stiffness tensor...
				if(tmp[k][l] < 0.)
				{
					if (me == 0) std::cout << "Carefull... Negative stiffness coefficient " << k << l << " - " << tmp[k][l] << std::endl;
					tmp[k][l] = -0.01*tmp[k][l];
					if (me == 0) std::cout << "Carefull... Replacing with " << tmp[k][l] << std::endl;
				}

			}

		// Write test... (on the data returned by lammps)

		// Conversion of the 6x6 Voigt Stiffness Tensor into the 3x3x3x3
		// Standard Stiffness Tensor
		for(unsigned int i=0;i<2*dim;i++)
		{
			int k, l;
			if     (i==(3+0)){k=1; l=2;}
			else if(i==(3+1)){k=0; l=2;}
			else if(i==(3+2)){k=0; l=1;}
			else  /*(i<3)*/  {k=i; l=i;}


			for(unsigned int j=0;j<2*dim;j++)
			{
				int m, n;
				if     (j==(3+0)){m=1; n=2;}
				else if(j==(3+1)){m=0; n=2;}
				else if(j==(3+2)){m=0; n=1;}
				else  /*(j<3)*/  {m=j; n=j;}

				stiffnesses[k][l][m][n]=tmp[i][j];

				//For debug...
				//stiffnesses[k][l][m][n] *= 0.1;
			}
		}

	}


	// The initiation, namely the preparation of the data from which will
	// be ran the later tests at every quadrature point, should be ran on
	// as many processes as available, since it will be the only on going
	// task at the time it will be called.
	template <int dim>
	void
	lammps_initiation (SymmetricTensor<4,dim>& initial_stress_strain_tensor,
					   MPI_Comm comm_lammps,
					   char* statelocin,
					   char* statelocout,
					   char* logloc)
	{
		// Compute init state even if available (true) or only if already absent (false);
		bool compute_state = false;

		// Locations for finding reference LAMMPS files, to store nanostate binary data, and
		// to place LAMMPS log/dump/temporary restart outputs
		char location[1024] = "../box";

		// Name of the nanostate binary file
		char initdata[1024] = "PE_init_end.bin";

		std::vector<std::vector<double> > tmp (2*dim, std::vector<double>(2*dim));

		int me;
		MPI_Comm_rank(comm_lammps, &me);

		// Repositories creation and checking...
		char qplogloc[1024];
		sprintf(qplogloc, "%s/%s", logloc, "init");
		mkdir(qplogloc, ACCESSPERMS);

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
		sprintf(lmparg[4], "%s/log.PE_heatup_cooldown", qplogloc);

		/*int nargs = 3;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-log";
		lmparg[2] = new char[1024];
		sprintf(lmparg[2], "%s/log.PE_heatup_cooldown", qplogloc);*/


		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,comm_lammps);

		// Passing location for input and output as variables
		sprintf(cline, "variable locb string %s", location); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", qplogloc); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps"); lammps_file(lmp,cfile);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal 200.0"); lammps_command(lmp,cline);
		sprintf(cline, "variable sseed equal 1111"); lammps_command(lmp,cline);

		// Check if 'PE_init_end.bin' has been computed already
		sprintf(sfile, "%s/%s", statelocin, initdata);
		bool state_exists = file_exists(sfile);

		if (compute_state || !state_exists)
		{
			if (me == 0) std::cout << "(MD - init) "
					<< "Compute state data...       " << std::endl;
			// Compute initialization of the sample which minimizes the free energy,
			// heat up and finally cool down the sample.
			sprintf(cfile, "%s/%s", location, "in.init.lammps"); lammps_file(lmp,cfile);
		}
		else
		{
			if (me == 0) std::cout << "(MD - init) "
					<< "Reuse of state data...       " << std::endl;
			// Reload from previously computed initial preparation (minimization and
			// heatup/cooldown), this option shouldn't remain, as in the first step the
			// preparation should always be computed.
			sprintf(cline, "read_restart %s/%s", statelocin, initdata); lammps_command(lmp,cline);
		}

		if (me == 0) std::cout << "(MD - init) "
				<< "Saving state data...       " << std::endl;
		sprintf(cline, "write_restart %s/%s", statelocout, initdata); lammps_command(lmp,cline);

		if (me == 0) std::cout << "(MD - init) "
				<< "Compute state using in.elastic.lammps...       " << std::endl;

		// Compute tangent stiffness operator
		SymmetricTensor<2,dim> stresses;
		lammps_state<dim>(lmp, location, stresses, initial_stress_strain_tensor);

		// close down LAMMPS
		delete lmp;
	}


	// The local_testing function is ran on every quadrature point which
	// requires a stress_update. Since a quandrature point is only reached*
	// by a subset of processes N, we should automatically see lammps be
	// parallelized on the N processes.
	template <int dim>
	void
	lammps_local_testing (const SymmetricTensor<2,dim>& strains,
			SymmetricTensor<2,dim>& stresses,
			SymmetricTensor<4,dim>& stress_strain_tensor,
			char* qptid,
			char* timeid,
			char* prev_timeid,
			MPI_Comm comm_lammps,
			char* statelocin,
			char* statelocout,
			char* logloc)
	{
		// Compute current state even if available (true) or only if already absent (false);
		// The choice of whether reuing a previous state or not should be done outside, if it has been computed
		// the macrostate as well has been stored so no need to restart the simulation just to homogenize...
		bool compute_state = true;

		// Locations for finding reference LAMMPS files, to store nanostate binary data, and
		// to place LAMMPS log/dump/temporary restart outputs
		char location[1024] = "../box";

		// Name of nanostate binary files
		char initdata[1024] = "PE_init_end.bin";
		char strainstate[1024] = "PE_strain_end.bin";

		std::vector<std::vector<double> > tmp (2*dim, std::vector<double>(2*dim));

		int me;
		MPI_Comm_rank(comm_lammps, &me);

		char qplogloc[1024];
		sprintf(qplogloc, "%s/%s.%s", logloc, timeid, qptid);
		mkdir(qplogloc, ACCESSPERMS);

		char straindata[1024];
		sprintf(straindata, "%s.%s.%s", timeid, qptid, strainstate);
		char straindata_old[1024];
		sprintf(straindata_old, "%s.%s.%s", prev_timeid, qptid, strainstate);

		char cline[1024];
		char cfile[1024];
		char mfile[1024];
		char sfile[1024];

		int nts;
		double dts;

		// Specifying the command line options for screen and log output file
		int nargs = 5;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-screen";
		lmparg[2] = (char *) "none";
		lmparg[3] = (char *) "-log";
		lmparg[4] = new char[1024];
		sprintf(lmparg[4], "%s/log.PE_stress_strain", qplogloc);

		/*int nargs = 3;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-log";
		lmparg[2] = new char[1024];
		sprintf(lmparg[2], "%s/log.PE_stress_strain", qpoutloc);*/


		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,comm_lammps);

		// Passing location for output as variable
		sprintf(cline, "variable loco string %s", qplogloc); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal 200.0"); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", location, "in.set.lammps");
		lammps_file(lmp,cfile);

		// Check if 'qptid.PE_strain_end.bin' has been computed already
		sprintf(sfile, "%s/%s", statelocin, straindata);
		bool state_exists = file_exists(sfile);

		if(compute_state || !state_exists)
		{
			if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
					<< "Compute current state data...       " << std::endl;
			// Compute from the initial state (true) or the previous state (false)
			bool compute_finit = true;
			// v_sound in PE is 2000m/s, since l0 = 4nm, with dts = 2.0fs, the condition
			// is nts > 1000 * strain so that v_load < v_sound...
			// Declaration of run parameters
			dts = 2.0; // timestep length in fs
			nts = 100; // number of timesteps

			// Set initial state of the testing box (either from initial end state
			// or from previous testing end state).
			if(compute_finit)
			{
				if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
						<< "   ... from init state data...       " << std::endl;
				sprintf(mfile, "%s/%s", statelocout, initdata);
			}
			else
			{
				if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
						<< "   ... from previous state data...   " << std::endl;
				sprintf(mfile, "%s/%s", statelocout, straindata_old);
			}

			std::ifstream ifile(mfile);
			if (!ifile.good()) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
					<< "Unable to open init/prev state file to read" << std::endl;

			sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

			sprintf(cline, "variable dts equal %f", dts); lammps_command(lmp,cline);
			sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

			for(unsigned int k=0;k<dim;k++)
				for(unsigned int l=k;l<dim;l++)
				{
					sprintf(cline, "variable eeps_%d%d equal %.6e", k, l, strains[k][l]/(nts*dts));
					lammps_command(lmp,cline);
				}

			// Run the NEMD simulations of the strained box
			if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
					<< "   ...reading and executing in.strain.lammps       " << std::endl;
			sprintf(cfile, "%s/%s", location, "in.strain.lammps");
			lammps_file(lmp,cfile);
		}
		else
		{
			if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
					<< "Reuse of current state data...       " << std::endl;

			sprintf(mfile, "%s/%s", statelocout, straindata);
			std::ifstream ifile(mfile);
			if (!ifile.good()) std::cout << "(" << timeid <<"."<< qptid << ") "
					<< "Unable to open strain_state file to read" << std::endl;
			sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);
		}


		if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
				<< "Saving state data...       " << std::endl;
		// Save data to specific file for this quadrature point
		sprintf(cline, "write_restart %s/%s", statelocout, straindata); lammps_command(lmp,cline);

		if (me == 0) std::cout << "(MD - " << timeid <<"."<< qptid << ") "
				<< "Compute state using in.elastic.lammps...       " << std::endl;

		// Compute the Tangent Stiffness Tensor at the given stress/strain state
		sprintf(cline, "variable nssample0 equal 100"); lammps_command(lmp,cline);
		lammps_state<dim>(lmp, location, stresses, stress_strain_tensor);

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
		const double rho = 7700;

		values = 0;
		values(dim-1) = -rho * g * 00000.;
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
	class IncrementalBoundaryValues :  public Function<dim>
	{
	public:
		IncrementalBoundaryValues (const double present_time,
				const double present_timestep);

		virtual
		void
		vector_value (const Point<dim> &p,
				Vector<double>   &values) const;

		virtual
		void
		vector_value_list (const std::vector<Point<dim> > &points,
				std::vector<Vector<double> >   &value_list) const;

	private:
		const double velocity;
		const double present_time;
		const double present_timestep;
	};


	template <int dim>
	IncrementalBoundaryValues<dim>::
	IncrementalBoundaryValues (const double present_time,
			const double present_timestep)
			:
			Function<dim> (dim),
			velocity (-0.001),
			present_time (present_time),
			present_timestep (present_timestep)
	{}


	template <int dim>
	void
	IncrementalBoundaryValues<dim>::
	vector_value (const Point<dim> &/*p*/,
			Vector<double>   &values) const
	{
		Assert (values.size() == dim,
				ExcDimensionMismatch (values.size(), dim));

		// All parts of the vector values are initiated to the given scalar.
		values = present_timestep * velocity;
	}



	template <int dim>
	void
	IncrementalBoundaryValues<dim>::
	vector_value_list (const std::vector<Point<dim> > &points,
			std::vector<Vector<double> >   &value_list) const
	{
		const unsigned int n_points = points.size();

		Assert (value_list.size() == n_points,
				ExcDimensionMismatch (value_list.size(), n_points));

		for (unsigned int p=0; p<n_points; ++p)
			IncrementalBoundaryValues<dim>::vector_value (points[p],
					value_list[p]);
	}



	template <int dim>
	class FEProblem
	{
	public:
		FEProblem (MPI_Comm dcomm, int pcolor, char* mslocin, char* mslocout, char* mlogloc);
		~FEProblem ();

		void make_grid ();
		void setup_system ();
		void restart_system ();
		void set_boundary_values (const double present_time, const double present_timestep);
		void assemble_system ();
		void solve_linear_problem ();
		void error_estimation ();
		double determine_step_length () const;
		void move_mesh ();

		void setup_quadrature_point_history ();

		void update_strain_quadrature_point_history
		(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no);
		void update_stress_quadrature_point_history
		(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no);

		void output_results (const double present_time, const int timestep_no) const;
		void output_state () const;

		double compute_residual () const;

		Vector<double> 		     			newton_update;
		Vector<double> 		     			incremental_displacement;
		Vector<double> 		     			solution;

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

		char*                                macrostatelocin;
		char*                                macrostatelocout;
		char*                                macrologloc;
	};



	template <int dim>
	FEProblem<dim>::FEProblem (MPI_Comm dcomm, int pcolor, char* mslocin, char* mslocout, char* mlogloc)
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
		macrostatelocin (mslocin),
		macrostatelocout (mslocout),
		macrologloc (mlogloc)
	{}



	template <int dim>
	FEProblem<dim>::~FEProblem ()
	{
		dof_handler.clear ();
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

		SymmetricTensor<4,dim> stiffness_tensor;
		char filename[1024];
		sprintf(filename, "%s/init.stiff", macrostatelocout);
		read_tensor<dim>(filename, stiffness_tensor);

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
					local_quadrature_points_history[q].new_stiff = stiffness_tensor;
					local_quadrature_points_history[q].new_stress = 0;
				}
			}

	}



	template <int dim>
	void FEProblem<dim>::update_strain_quadrature_point_history
	(const Vector<double>& displacement_update, const int timestep_no, const int newtonstep_no)
	{
		char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);

		// Create file with qptid to update at timeid
		std::ofstream ofile;
		char update_local_filename[1024];
		sprintf(update_local_filename, "%s/%s.%d.qpupdates", macrostatelocout, time_id, this_FE_process);
		ofile.open (update_local_filename);

		// Preparing requirements for strain update
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		double strain_perturbation = 0.005;

		if (newtonstep_no > 0) dcout << "        " << "...checking quadrature points requiring update..." << std::endl;

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
				fe_values.get_function_gradients (displacement_update,
						displacement_update_grads);

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					local_quadrature_points_history[q].to_be_updated = false;

					local_quadrature_points_history[q].old_strain =
							local_quadrature_points_history[q].new_strain;

					local_quadrature_points_history[q].old_stress =
							local_quadrature_points_history[q].new_stress;

					local_quadrature_points_history[q].old_stiff =
							local_quadrature_points_history[q].new_stiff;

					// Strain tensor update
					local_quadrature_points_history[q].new_strain +=
							get_strain (displacement_update_grads[q]);

					local_quadrature_points_history[q].inc_strain =
							get_strain (displacement_update_grads[q]);

					local_quadrature_points_history[q].upd_strain +=
							get_strain (displacement_update_grads[q]);

					// CREATE FILES FOR STRAIN STORAGE ONLY FOR QPT TO BE UPDATED.
					// Might want to create a more robust way of listing points, using list of points
					// to be updated
					// Share this list to all procs either using file dump or collective
					//  	 > Use PETScWrappers tools d, as described in tutorials, they
					// 		   can be used since data shared is vector of vector of bool/int.

					// Store the cumulative strain since last update, if one of components is above
					// the 'strain perturbation' (curr. 0.005%) used in LAMMPS to compute the tangent
					// linear stiffness, declare the quadrature point to be updated and reset the
					// cummulative strain
					// For debug...
					/*if (//false
							(cell->active_cell_index() == 10)
							//or (cell->active_cell_index() == 3 && (q == 2))
							)
						for(unsigned int k=0;k<dim;k++){
							for(unsigned int l=k;l<dim;l++) std::cout << local_quadrature_points_history[q].upd_strain[k][l] << " ";
							std::cout << std::endl;
						}*/

					/*if (//false
						(cell->active_cell_index() == 21 || cell->active_cell_index() == 12
								|| cell->active_cell_index() == 10 || cell->active_cell_index() == 5)
						) // For debug... */
					if (newtonstep_no > 0)
						for(unsigned int k=0;k<dim;k++){
							for(unsigned int l=k;l<dim;l++){
//								std::cout << local_quadrature_points_history[q].upd_strain[k][l] << std::endl;
								if (fabs(local_quadrature_points_history[q].upd_strain[k][l]) > strain_perturbation
										&& local_quadrature_points_history[q].to_be_updated == false){
									std::cout << "           "
											<< " cell "<< cell->active_cell_index() << " QP " << q
											<< " strain component " << k << l
											<< " value " << local_quadrature_points_history[q].upd_strain[k][l] << std::endl;

									local_quadrature_points_history[q].to_be_updated = true;
									local_quadrature_points_history[q].upd_strain = 0;
								}
							}
//							if(k==dim-1) std::cout << "****" << std::endl;
						}

					// Write strain and previous stiffness tensors in case the quadrature point needs to be updated
					if (local_quadrature_points_history[q].to_be_updated){
						// Write total strains in a file named ./macrostate_storage/time.it-cellid.qid.strain
						char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
						char filename[1024];

						sprintf(filename, "%s/%s.%s.strain", macrostatelocout, time_id, quad_id);
						write_tensor<dim>(filename, local_quadrature_points_history[q].new_strain);

						// For debug...
						char prev_time_id[1024]; sprintf(prev_time_id, "%d-%d", timestep_no, newtonstep_no-1);
						sprintf(filename, "%s/%s.%s.stiff", macrostatelocout, prev_time_id, quad_id);
						write_tensor<dim>(filename, local_quadrature_points_history[q].old_stiff);

						ofile << quad_id << std::endl;
					}

				}
			}
		ofile.close();
		MPI_Barrier(FE_communicator);

		// Gathering in a single file all the quadrature points to be updated...
		// Might be worth replacing indivual local file writings by a parallel vector of string
		// and globalizing this vector before this final writing step.
		std::ifstream ifile;
		std::ofstream outfile;
		std::string iline;
		if (this_FE_process == 0){
			char update_filename[1024];
			sprintf(update_filename, "%s/%s.qpupdates", macrostatelocout, time_id);
			outfile.open (update_filename);
			for (int ip=0; ip<n_FE_processes; ip++){
				sprintf(update_local_filename, "%s/%s.%d.qpupdates", macrostatelocout, time_id, ip);
				ifile.open (update_local_filename);
				while (getline(ifile, iline)) outfile << iline << std::endl;
				ifile.close();
			}
			outfile.close();
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


		// Retrieving all quadrature points computation and storing them in the
		// quadrature_points_history structure
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
				fe_values.get_function_gradients (displacement_update,
						displacement_update_grads);

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					// Restore the new stress and stiffness tensors from two files, respectively
					// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
					char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);
					char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
					char filename[1024];

//					sprintf(filename, "%s/%s.%s.stress", storloc, time_id, quad_id);
//					read_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);
					if (local_quadrature_points_history[q].to_be_updated){
						sprintf(filename, "%s/%s.%s.stiff", macrostatelocout, time_id, quad_id);
						read_tensor<dim>(filename, local_quadrature_points_history[q].new_stiff);
					}

					// Secant stiffness computation of the new stress tensor
					local_quadrature_points_history[q].new_stress =
							local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].new_strain;

					// Apply rotation of the sample to the new state tensors
					const Tensor<2,dim> rotation
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

					local_quadrature_points_history[q].new_stress
					= rotated_new_stress;
					local_quadrature_points_history[q].new_strain
					= rotated_new_strain;
				}
			}
	}



	template <int dim>
	void FEProblem<dim>::set_boundary_values
	(const double present_time, const double present_timestep)
	{
		FEValuesExtractors::Scalar t_component (dim-3);
		FEValuesExtractors::Scalar h_component (dim-2);
		FEValuesExtractors::Scalar v_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;

		VectorTools::
		interpolate_boundary_values (dof_handler,
				11,
				ZeroFunction<dim>(dim),
				boundary_values);

		VectorTools::
		interpolate_boundary_values (dof_handler,
				12,
				ZeroFunction<dim>(dim),
				boundary_values);

		VectorTools::
		interpolate_boundary_values (dof_handler,
				12,
				IncrementalBoundaryValues<dim>(present_time,
						present_timestep),
				boundary_values,
				fe.component_mask(t_component));

		for (std::map<types::global_dof_index, double>::const_iterator
				p = boundary_values.begin();
				p != boundary_values.end(); ++p)
			incremental_displacement(p->first) = p->second;
	}



	template <int dim>
	void FEProblem<dim>::assemble_system ()
	{
		system_rhs = 0;
		system_matrix = 0;

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

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
			if (cell->is_locally_owned())
			{
				cell_matrix = 0;
				cell_rhs = 0;

				fe_values.reinit (cell);

				const PointHistory<dim> *local_quadrature_points_data
				= reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

				for (unsigned int i=0; i<dofs_per_cell; ++i)
					for (unsigned int j=0; j<dofs_per_cell; ++j)
						for (unsigned int q_point=0; q_point<n_q_points;
								++q_point)
						{
							const SymmetricTensor<4,dim> &new_stiff
							= local_quadrature_points_data[q_point].new_stiff;

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
						const SymmetricTensor<2,dim> &new_stress
						= local_quadrature_points_data[q_point].new_stress;

						cell_rhs(i) += (body_force_values[q_point](component_i) *
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

		FEValuesExtractors::Scalar t_component (dim-3);
		FEValuesExtractors::Scalar h_component (dim-2);
		FEValuesExtractors::Scalar v_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;

		VectorTools::
		interpolate_boundary_values (dof_handler,
				11,
				ZeroFunction<dim>(dim),
				boundary_values);

		VectorTools::
		interpolate_boundary_values (dof_handler,
				12,
				ZeroFunction<dim>(dim),
				boundary_values);

		VectorTools::
		interpolate_boundary_values (dof_handler,
				12,
				ZeroFunction<dim>(dim),
				boundary_values,
				fe.component_mask(t_component));

		PETScWrappers::MPI::Vector tmp (locally_owned_dofs,FE_communicator);
		MatrixTools::apply_boundary_values (boundary_values,
				system_matrix,
				tmp,
				system_rhs,
				false);
		newton_update = tmp;

		dcout << "    FE System - norm of rhs is " << system_rhs.l2_norm()
							  << std::endl;


	}



	template <int dim>
	void FEProblem<dim>::solve_linear_problem ()
	{
		PETScWrappers::MPI::Vector
		distributed_newton_update (locally_owned_dofs,FE_communicator);
		distributed_newton_update = newton_update;

		SolverControl       solver_control (1000,
				1e-16*system_rhs.l2_norm());
		PETScWrappers::SolverCG cg (solver_control,
				FE_communicator);

		// Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
		// not optimal for large scale simulations.
		PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
		cg.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update);

		const double alpha = determine_step_length();
		incremental_displacement.add (alpha, newton_update);

		dcout << "    FE Solver - norm of newton update is " << newton_update.l2_norm()
							  << std::endl;
		dcout << "    FE Solver converged in " << solver_control.last_step()
				<< " iterations." << std::endl;
	}



	template <int dim>
	double FEProblem<dim>::compute_residual () const
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

		BodyForce<dim>      body_force;
		std::vector<Vector<double> > body_force_values (n_q_points,
				Vector<double>(dim));

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
			if (cell->is_locally_owned())
			{
				cell_residual = 0;
				fe_values.reinit (cell);

				const PointHistory<dim> *local_quadrature_points_data
				= reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
				body_force.vector_value_list (fe_values.get_quadrature_points(),
						body_force_values);

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					const unsigned int
					component_i = fe.system_to_component_index(i).first;

					for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
					{
						const SymmetricTensor<2,dim> &old_stress
						= local_quadrature_points_data[q_point].new_stress;

						cell_residual(i) += (body_force_values[q_point](component_i) *
								fe_values.shape_value (i,q_point)
								-
								old_stress *
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

		// This manner to remove lines concerned with boundary conditions in the
		// residual vector does not yield the same norm for the residual vector
		// and the system_rhs vector (for which boundary conditions are applied
		// differently).
		// Is the value obtained with this method correct?
		// Should we proceed differently to obtain the same norm value? Although
		// localizing the vector (see step-17) does not change the norm value.
		std::vector<bool> boundary_dofs (dof_handler.n_dofs());
		DoFTools::extract_boundary_dofs (dof_handler,
				ComponentMask(),
				boundary_dofs);
		for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
			if (boundary_dofs[i] == true)
				residual(i) = 0;

		return residual.l2_norm();
	}



	template <int dim>
	void FEProblem<dim>::error_estimation ()
	{
		error_per_cell.reinit (triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate (dof_handler,
				QGauss<dim-1>(2),
				typename FunctionMap<dim>::type(),
				newton_update,
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
	void FEProblem<dim>::output_results (const double present_time, const int timestep_no) const
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler (dof_handler);

		// Output of displacement as a vector
		std::vector<std::string>  solution_names (dim, "displacement");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation
		(dim, DataComponentInterpretation::component_is_part_of_vector);
		data_out.add_data_vector (solution,
				solution_names,
				DataOut<dim>::type_dof_data,
				data_component_interpretation);

		// Output of error per cell as a scalar
		data_out.add_data_vector (error_per_cell, "error_per_cell");

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
			data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
			//DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

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
			data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
			//DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
		}
	}



	template <int dim>
	void FEProblem<dim>::output_state () const
	{
		if (this_FE_process==0)
		{
			// Write solution vector to binary for simulation restart
			std::string smacrostatelocout(macrostatelocout);
			const std::string solution_filename = (smacrostatelocout + "/" + "last.solution.bin");
			std::ofstream ofile(solution_filename);
			solution.block_write(ofile);
			ofile.close();
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

					for (unsigned int q=0; q<quadrature_formula.size(); ++q)
					{
						char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
						char filename[1024];

						// Write stress tensor to human readable format file for simulation restart
						sprintf(filename, "%s/last.%s.stress", macrostatelocout, quad_id);
						write_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);
					}
				}*/
	}



	template <int dim>
	void FEProblem<dim>::make_grid ()
	{
		std::vector< unsigned int > sizes (GeometryInfo<dim>::faces_per_cell);
		sizes[0] = 0; sizes[1] = 1;
		sizes[2] = 0; sizes[3] = 0;
		sizes[4] = 0; sizes[5] = 0;
		GridGenerator::hyper_cross(triangulation, sizes);
		for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
				cell != triangulation.end();
				++cell)
			for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
				if (cell->face(f)->at_boundary())
				{
					if (cell->face(f)->center()[0] == -0.5)
						cell->face(f)->set_boundary_id (11);
					if (cell->face(f)->center()[0] == 1.5)
						cell->face(f)->set_boundary_id (12);
					if (cell->face(f)->center()[1] == -0.5)
						cell->face(f)->set_boundary_id (21);
					if (cell->face(f)->center()[1] == 0.5)
						cell->face(f)->set_boundary_id (22);
					if (cell->face(f)->center()[2] == -0.5)
						cell->face(f)->set_boundary_id (31);
					if (cell->face(f)->center()[2] == 0.5)
						cell->face(f)->set_boundary_id (32);
				}
		triangulation.refine_global (2);

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

		incremental_displacement.reinit (dof_handler.n_dofs());
		newton_update.reinit (dof_handler.n_dofs());
		solution.reinit (dof_handler.n_dofs());

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
	void FEProblem<dim>::restart_system ()
	{
		char filename[1024];

		dcout << "    initialization of the position vector... " << std::endl;
		sprintf(filename, "%s/last.solution.bin", macrostatelocin);
		std::ifstream ifile(filename);
		if (ifile.is_open())
		{
			dcout << "       ...from previous simulation " << std::endl;
			solution.block_read(ifile);
			ifile.close();
		}
		else dcout << "       ...to zero " << std::endl;

		/*dcout << "  Initialization of the local stress tensors... " << std::endl;
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
						char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);

						// Write stress tensor to human readable format file for simulation restart
						sprintf(filename, "%s/last.%s.stress", macrostatelocin, quad_id);

						ifile.open (filename);
						if (ifile.is_open())
						{
							read_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);
						}
					}
				}*/
	}





	template <int dim>
	class HMMProblem
	{
	public:
		HMMProblem ();
		~HMMProblem ();
		void run ();

	private:
		void set_repositories ();
		void set_dealii_procs ();
		void set_lammps_procs ();
		void initial_stiffness_with_molecular_dynamics ();
		void do_timestep (FEProblem<dim> &fe_problem);
		void solve_timestep (FEProblem<dim> &fe_problem);

		void update_stiffness_with_molecular_dynamics ();

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		MPI_Comm 							dealii_communicator;
		int									root_dealii_process;
		int 								n_dealii_processes;
		int 								this_dealii_process;
		int 								dealii_pcolor;

		MPI_Comm 							lammps_global_communicator;
		MPI_Comm 							lammps_batch_communicator;
		int 								n_lammps_processes;
		int 								n_lammps_processes_per_batch;
		int 								n_lammps_batch;
		int 								this_lammps_process;
		int 								this_lammps_batch_process;
		int 								lammps_pcolor;

		ConditionalOStream 					hcout;

		double              				present_time;
		double              				present_timestep;
		double              				end_time;
		int        							timestep_no;
		int        							newtonstep_no;

		SymmetricTensor<4,dim> 				initial_stress_strain_tensor;

		char                                macrostateloc[1024];
		char                                macrostatelocin[1024];
		char                                macrostatelocout[1024];
		char                                macrologloc[1024];

		char                                nanostateloc[1024];
		char                                nanostatelocin[1024];
		char                                nanostatelocout[1024];
		char                                nanologloc[1024];

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
	void HMMProblem<dim>::update_stiffness_with_molecular_dynamics()
	{
		char prev_time_id[1024]; sprintf(prev_time_id, "%d-%d", timestep_no, newtonstep_no-1);
		char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);

		// Check list of files corresponding to current "time_id"
		int nqupd = 0;
		char filenamelist[1024];
		sprintf(filenamelist, "%s/%s.qpupdates", macrostatelocout, time_id);
		std::ifstream ifile;
		std::string iline;

		// Count number of quadrature point to update
		ifile.open (filenamelist);
		if (ifile.is_open())
		{
		    while (getline(ifile, iline)) nqupd++;
			ifile.close();
		}
		else hcout << "Unable to open" << filenamelist << " to read it" << std::endl;

		// Create list of quadid
		char **quad_id = new char *[nqupd];
		for (int q=0; q<nqupd; ++q) quad_id[q] = new char[1024];

		ifile.open (filenamelist);
		int nline = 0;
		while (nline<nqupd && ifile.getline(quad_id[nline], sizeof(quad_id[nline]))) nline++;
		ifile.close();

		// It might be worth doing the splitting of in batches of lammps processors here according to
		// the number of quadrature points to update, because if the number of points is smaller than
		// the number of batches predefined initially part of the lammps allocated processors remain idle...
		hcout << "        " << "...dispatching the MD runs on batch of processes..." << std::endl;
		for (int q=0; q<nqupd; ++q)
		{
			if (lammps_pcolor == (q%n_lammps_batch))
			{
				SymmetricTensor<2,dim> loc_strain;

				// Restore the strain tensor from the file ./macrostate_storage/time.it-cellid.qid.strain
				//				char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
				char filename[1024];

				sprintf(filename, "%s/%s.%s.strain", macrostatelocout, time_id, quad_id[q]);
				read_tensor<dim>(filename, loc_strain);

				SymmetricTensor<2,dim> loc_stress;
				SymmetricTensor<4,dim> loc_stiffness;

				// For debug...
				int me;
				MPI_Comm_rank(lammps_batch_communicator, &me);
				std::cout << "            "
						<< "nqptbu: " << q
						<< " - cell - qp : " << quad_id[q]
						<< " - proc_world_rank: " << this_lammps_process
						<< " - lammps batch computed: " << (q%n_lammps_batch)
						<< " - lammps batch color: " << lammps_pcolor
						<< " - proc_batch_rank: " << me
						<< std::endl;

				// For debug...
				sprintf(filename, "%s/%s.%s.stiff", macrostatelocout, prev_time_id, quad_id[q]);
				read_tensor<dim>(filename, loc_stiffness);

				// For debug...
				if(this_lammps_batch_process == 0)
				{
					std::cout << "Old Stiffnesses: "<< loc_stiffness[0][0][0][0]
													<< " " << loc_stiffness[1][1][1][1]
													<< " " << loc_stiffness[2][2][2][2] << " " << std::endl;
				}

				// Then the lammps function instanciates lammps, starting from an initial
				// microstructure and applying the complete new_strain or starting from
				// the microstructure at the old_strain and applying the difference between
				// the new_ and _old_strains, returns the new_stress state.
//				lammps_local_testing<dim> (loc_strain,
//						loc_stress,
//						loc_stiffness,
//						quad_id[q],
//						time_id,
//						prev_time_id,
//						lammps_batch_communicator,
//						nanostatelocin,
//						nanostatelocout,
//						nanologloc);

				// For debug...
				sprintf(filename, "%s/%s.%s.stiff", macrostatelocout, prev_time_id, quad_id[q]);
				read_tensor<dim>(filename, loc_stiffness);
				// For debug...
				for (unsigned int i=0; i<dim; ++i)
					for (unsigned int j=0; j<dim; ++j)
						for (unsigned int k=0; k<dim; ++k)
							for (unsigned int l=0; l<dim; ++l)
								loc_stiffness[i][j][k][l] *= 0.90;

				// Write the new stress and stiffness tensors into two files, respectively
				// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
				if(this_lammps_batch_process == 0)
				{
					// For debug...
					std::cout << "Stiffnesses: "<< loc_stiffness[0][0][0][0]
												<< " " << loc_stiffness[1][1][1][1]
												<< " " << loc_stiffness[2][2][2][2] << " " << std::endl;

					//							sprintf(filename, "%s/%s.%s.stress", storloc, time_id, quad_id);
					//							write_tensor<dim>(filename, loc_stress);

					sprintf(filename, "%s/%s.%s.stiff", macrostatelocout, time_id, quad_id[q]);
					write_tensor<dim>(filename, loc_stiffness);
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
			if(dealii_pcolor==0) previous_res = fe_problem.compute_residual();
			hcout << "  Initial residual: "
					<< previous_res
					<< std::endl;

			for (unsigned int inner_iteration=0; inner_iteration<5; ++inner_iteration)
			{
				++newtonstep_no;
				hcout << "    Assembling FE system..." << std::flush;
				if(dealii_pcolor==0) fe_problem.assemble_system ();

				hcout << "    Solving FE system..." << std::flush;
				if(dealii_pcolor==0) fe_problem.solve_linear_problem ();

				hcout << "    Updating quadrature point data..." << std::endl;

				if(dealii_pcolor==0) fe_problem.update_strain_quadrature_point_history
						(fe_problem.newton_update, timestep_no, newtonstep_no);
				MPI_Barrier(world_communicator);

				if(lammps_pcolor>=0) update_stiffness_with_molecular_dynamics();
				MPI_Barrier(world_communicator);

				if(dealii_pcolor==0) fe_problem.update_stress_quadrature_point_history
						(fe_problem.newton_update, timestep_no, newtonstep_no);

				if(dealii_pcolor==0) previous_res = fe_problem.compute_residual();
				MPI_Barrier(world_communicator);

				// Share the value of previous_res in between processors
				MPI_Bcast(&previous_res, 1, MPI_DOUBLE, root_dealii_process, world_communicator);

				hcout << "  Residual: "
						<< previous_res
						<< std::endl;
			}
		} while (previous_res>1e-3);
	}




	template <int dim>
	void HMMProblem<dim>::do_timestep (FEProblem<dim> &fe_problem)
	{

		present_time += present_timestep;
		++timestep_no;
		hcout << "Timestep " << timestep_no << " at time " << present_time
				<< std::endl;
		if (present_time > end_time)
		{
			present_timestep -= (present_time - end_time);
			present_time = end_time;
		}

		newtonstep_no = 0;

		if(dealii_pcolor==0) fe_problem.incremental_displacement = 0;

		if(dealii_pcolor==0) fe_problem.set_boundary_values (present_time, present_timestep);

		if(dealii_pcolor==0) fe_problem.update_strain_quadrature_point_history (fe_problem.incremental_displacement, timestep_no, newtonstep_no);
		MPI_Barrier(world_communicator);

		// At the moment, the stiffness is never updated here, due to checking condition
		// when updating strains (newtonstep_no < 0).
		// This is a safety check, because if load increment are important they force stiffness update,
		// because all loading is localized in cells next to imposed DOF.
		// When the loading increments will be reduced, the condition during the strain update can be removed
		// although it might never really be useful to update the stress at that time, because new_strains at
		// (newtonstep_no == 0) will always be quite far from the converged values..
		/*if(lammps_pcolor>=0) update_stiffness_with_molecular_dynamics();
		//MPI_Barrier(world_communicator);*/

		if(dealii_pcolor==0) fe_problem.update_stress_quadrature_point_history (fe_problem.incremental_displacement, timestep_no, newtonstep_no);

		solve_timestep (fe_problem);

		if(dealii_pcolor==0) fe_problem.solution+=fe_problem.incremental_displacement;

		if(dealii_pcolor==0) fe_problem.error_estimation ();

		if(dealii_pcolor==0) fe_problem.output_results (present_time, timestep_no);

		if(dealii_pcolor==0) fe_problem.output_state ();

		hcout << std::endl;
	}




	template <int dim>
	void HMMProblem<dim>::initial_stiffness_with_molecular_dynamics ()
	{
		char macrofilenamein[1024];
		sprintf(macrofilenamein, "%s/init.stiff", macrostatelocin);
		char macrofilenameout[1024];
		sprintf(macrofilenameout, "%s/init.stiff", macrostatelocout);
		bool macrostate_exists = file_exists(macrofilenamein);

		char nanofilenamein[1024];
		sprintf(nanofilenamein, "%s/PE_init_end.bin", nanostatelocin);
		char nanofilenameout[1024];
		sprintf(nanofilenameout, "%s/PE_init_end.bin", nanostatelocout);
		bool nanostate_exists = file_exists(nanofilenamein);

		if(!macrostate_exists || !nanostate_exists){
			hcout << " ...from a molecular dynamics simulation       " << std::endl;
//			if(lammps_pcolor>=0) lammps_initiation<dim> (initial_stress_strain_tensor, lammps_global_communicator,
//					                                     nanostatelocin, nanostatelocout, nanologloc);

			// For debug... using arbitrary stiffness tensor...
			if(this_lammps_process == 0){
				double young = 3.0e9, poisson = 0.45;
				double mu = 0.5*young/(1+poisson), lambda = young*poisson/((1+poisson)*(1-2*poisson));
				for (unsigned int i=0; i<dim; ++i)
					for (unsigned int j=0; j<dim; ++j)
						for (unsigned int k=0; k<dim; ++k)
							for (unsigned int l=0; l<dim; ++l)
								initial_stress_strain_tensor[i][j][k][l]
																	  = (((i==k) && (j==l) ? mu : 0.0) +
																			  ((i==l) && (j==k) ? mu : 0.0) +
																			  ((i==j) && (k==l) ? lambda : 0.0));
			}
			MPI_Barrier(world_communicator);


			if(this_lammps_process == 0) write_tensor<dim>(macrofilenameout, initial_stress_strain_tensor);
		}
		else{
			hcout << " ...from an existing stiffness tensor       " << std::endl;
			if(this_lammps_process == 0){
			    std::ifstream  macroin(macrofilenamein, std::ios::binary);
			    std::ofstream  macroout(macrofilenameout,   std::ios::binary);
			    macroout << macroin.rdbuf();
			    macroin.close();
			    macroout.close();

			    std::ifstream  nanoin(nanofilenamein, std::ios::binary);
			    std::ofstream  nanoout(nanofilenameout,   std::ios::binary);
			    nanoout << nanoin.rdbuf();
			    nanoin.close();
			    nanoout.close();
			}
		}
	}




	// There are several number of processes encountered: (i) n_lammps_processes the highest provided
	// as an argument to aprun, (ii) ND the number of processes provided to deal.ii
	// [arbitrary], (iii) NI the number of processes provided to the lammps initiation
	// [as close as possible to n_lammps_processes], and (iv) n_lammps_processes_per_batch the number of processes provided to one lammps
	// testing [NT divided by n_lammps_batch the number of concurrent testing boxes].
	template <int dim>
	void HMMProblem<dim>::set_lammps_procs ()
	{
		// Create a communicator for all processes allocated to lammps
		MPI_Comm_dup(MPI_COMM_WORLD, &lammps_global_communicator);

		MPI_Comm_rank(lammps_global_communicator,&this_lammps_process);
		MPI_Comm_size(lammps_global_communicator,&n_lammps_processes);

		// Arbitrary setting of NB and NT
		n_lammps_processes_per_batch = 2;

		n_lammps_batch = int(n_lammps_processes/n_lammps_processes_per_batch);
		if(n_lammps_batch == 0) {n_lammps_batch=1; n_lammps_processes_per_batch=n_lammps_processes;}

		lammps_pcolor = MPI_UNDEFINED;

		// LAMMPS processes color: regroup processes by batches of size NB, except
		// the last ones (me >= NB*NC) to create batches of only NB processes, nor smaller.
		if(this_lammps_process < n_lammps_processes_per_batch*n_lammps_batch)
			lammps_pcolor = int(this_lammps_process/n_lammps_processes_per_batch);

		// Definition of the communicators
		MPI_Comm_split(lammps_global_communicator, lammps_pcolor, this_lammps_process, &lammps_batch_communicator);
		MPI_Comm_rank(lammps_batch_communicator,&this_lammps_batch_process);
	}




	template <int dim>
	void HMMProblem<dim>::set_dealii_procs ()
	{
		root_dealii_process = 0;
		n_dealii_processes = 2;

		dealii_pcolor = MPI_UNDEFINED;

		// Color set above 0 for processors that are going to be used
		if (this_world_process >= root_dealii_process &&
				this_world_process < root_dealii_process + n_dealii_processes) dealii_pcolor = 0;
		else dealii_pcolor = 1;

		MPI_Comm_split(MPI_COMM_WORLD, dealii_pcolor, this_world_process, &dealii_communicator);
		MPI_Comm_rank(dealii_communicator, &this_dealii_process);
	}




	template <int dim>
	void HMMProblem<dim>::set_repositories ()
	{
		sprintf(macrostateloc, "./macroscale_state"); mkdir(macrostateloc, ACCESSPERMS);
		sprintf(macrostatelocin, "%s/in", macrostateloc); mkdir(macrostatelocin, ACCESSPERMS);
		sprintf(macrostatelocout, "%s/out", macrostateloc); mkdir(macrostatelocout, ACCESSPERMS);
		sprintf(macrologloc, "./macroscale_log"); mkdir(macrologloc, ACCESSPERMS);

		sprintf(nanostateloc, "./nanoscale_state"); mkdir(nanostateloc, ACCESSPERMS);
		sprintf(nanostatelocin, "%s/in", nanostateloc); mkdir(nanostatelocin, ACCESSPERMS);
		sprintf(nanostatelocout, "%s/out", nanostateloc); mkdir(nanostatelocout, ACCESSPERMS);
		sprintf(nanologloc, "./nanoscale_log"); mkdir(nanologloc, ACCESSPERMS);
	}



	template <int dim>
	void HMMProblem<dim>::run ()
	{
		// Setting repositories for input and creating repositories for outputs
		set_repositories();
		MPI_Barrier(world_communicator);

		hcout << "Building the HMM problem:       " << std::endl;
		// Set the dealii communicator using a limited amount of available processors
		// because dealii fails if processors do not have assigned cells. Plus, dealii
		// might not scale indefinitely
		set_dealii_procs();

		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		set_lammps_procs();

		// Recapitulating allocation of each process to deal and lammps
		std::cout << "proc world rank: " << this_world_process
				<< " - deal color: " << dealii_pcolor
				<< " - lammps color: " << lammps_pcolor << std::endl;

		// Construct FE class
		hcout << " Initiation of the Finite Element problem...       " << std::endl;
		FEProblem<dim> fe_problem (dealii_communicator, dealii_pcolor,
				                    macrostatelocin, macrostatelocout, macrologloc);
		MPI_Barrier(world_communicator);

		// Since LAMMPS is highly scalable, the initiation number of processes NI
		// can basically be equal to the maximum number of available processes NT which
		// can directly be found in the MPI_COMM.
		hcout << " Initialization of stiffness and initiation of the Molecular Dynamics sample...       " << std::endl;
		initial_stiffness_with_molecular_dynamics();
		MPI_Barrier(world_communicator);

		// Initialization of time variables
		present_time = 0;
		present_timestep = 1;
		end_time = 10;
		timestep_no = 0;

		hcout << " Initiation of the Mesh...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.make_grid ();

		hcout << " Initiation of the global vectors and tensor...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.setup_system ();

		hcout << " Initiation of the local tensors...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.setup_quadrature_point_history ();

		hcout << " Loading previous simulation data...       " << std::endl;
		if(dealii_pcolor==0) fe_problem.restart_system ();

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
