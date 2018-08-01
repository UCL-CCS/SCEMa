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

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "library.h"
#include "atom.h"

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
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
	struct ReplicaData
	{
		// Characteristics
		double rho;
		std::string mat;
		int repl;
		int nflakes;
		Tensor<1,dim> length;
		Tensor<2,dim> rotam;
		SymmetricTensor<2,dim> init_stress;
		SymmetricTensor<4,dim> init_stiffness;
	};

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
		Tensor<2,dim> rotam;
	};

    void bptree_print(boost::property_tree::ptree const& pt)
    {
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt.end();
        for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
            std::cout << it->first << ": " << it->second.get_value<std::string>() << std::endl;
            bptree_print(it->second);
        }
    }

    std::string bptree_read(boost::property_tree::ptree const& pt, std::string key)
    {
    	std::string value = "NULL";
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt.end();
        for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
            if(it->first==key)
            	value = it->second.get_value<std::string>();
        }
        return value;
    }

    std::string bptree_read(boost::property_tree::ptree const& pt, std::string key1, std::string key2)
    {
    	std::string value = "NULL";
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt.end();
        for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
            if(it->first==key1){
            	value = bptree_read(it->second, key2);
            }
        }
        return value;
    }

    std::string bptree_read(boost::property_tree::ptree const& pt, std::string key1, std::string key2, std::string key3)
    {
    	std::string value = "NULL";
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt.end();
        for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
            if(it->first==key1){
            	value = bptree_read(it->second, key2, key3);
            }
        }
        return value;
    }

    boost::property_tree::ptree get_subbptree(boost::property_tree::ptree const& pt, std::string key1)
    {
    	boost::property_tree::ptree value;
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt.end();
        for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
            if(it->first==key1){
            	value = it->second;
            }
        }
        return value;
    }


	bool file_exists(std::string file) {
		struct stat buf;
		return (stat(file.c_str(), &buf) == 0);
	}


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
	bool
	read_tensor (char *filename, SymmetricTensor<2,dim> &tensor)
	{
		std::ifstream ifile;

		bool load_ok = false;

		ifile.open (filename);
		if (ifile.is_open())
		{
			load_ok = true;
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
	return load_ok;
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
	Tensor<2,dim>
	compute_rotation_tensor (Tensor<1,dim> vorig, Tensor<1,dim> vdest)
	{
		Tensor<2,dim> rotam;

		// Filling identity matrix
		Tensor<2,dim> idmat;
		idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;

		// Decalaration variables rotation matrix computation
		double ccos;
		Tensor<2,dim> skew_rot;

		// Compute the scalar product of the local and global vectors
		ccos = scalar_product(vorig, vdest);

		// Filling the skew-symmetric cross product matrix (a^Tb-b^Ta)
		for (unsigned int i=0; i<dim; ++i)
			for (unsigned int j=0; j<dim; ++j)
				skew_rot[i][j] = vorig[j]*vdest[i] - vorig[i]*vdest[j];

		// Assembling the rotation matrix
		rotam = idmat + skew_rot + (1/(1+ccos))*skew_rot*skew_rot;

		return rotam;
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

		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				stmp[k][l] = 0.5*(tmp[k][l] + tmp[l][k]);

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
		values(dim-1) = -g;
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
		FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for,
				std::vector<std::string> mdtype, Tensor<1,dim> cg_dir);
		~FEProblem ();

		void init (int sstp, double tlength, std::string mslocin, std::string mslocout,
				   std::string mslocres, std::string mlogloc, int fchpt, int fovis, int folhis,
				   bool actmdup, std::vector<std::string> mdt, Tensor<1,dim> cgd);
		void beginstep (int tstp, double ptime);
		void solve (int nstp);
		bool check ();
		void endstep ();

	private:
		void make_grid ();
		void setup_system ();
		std::vector<Vector<double> > get_microstructure ();
		void assign_microstructure (typename DoFHandler<dim>::active_cell_iterator cell, std::vector<Vector<double> > structure_data,
				std::string &mat, Tensor<2,dim> &rotam);
		void setup_quadrature_point_history ();
		void restart ();

		void set_boundary_values ();

		double assemble_system (bool first_assemble);
		void solve_linear_problem_CG ();
		void solve_linear_problem_GMRES ();
		void solve_linear_problem_BiCGStab ();
		void solve_linear_problem_direct ();
		void update_incremental_variables ();
		void update_strain_quadrature_point_history
		(const Vector<double>& displacement_update);

		void update_stress_quadrature_point_history
		(const Vector<double>& displacement_update,);
		void clean_transfer();

		Vector<double>  compute_internal_forces () const;
		void output_lhistory ();
		void output_visualisation ();
		void output_results ();
		void checkpoint (char* timeid) const;

		Vector<double> 		     			newton_update_displacement;
		Vector<double> 		     			incremental_displacement;
		Vector<double> 		     			displacement;
		Vector<double> 		     			old_displacement;

		Vector<double> 		     			newton_update_velocity;
		Vector<double> 		     			incremental_velocity;
		Vector<double> 		     			velocity;
		//Vector<double> 		     		old_velocity;

		MPI_Comm 							FE_communicator;
		int 								n_FE_processes;
		int 								this_FE_process;
		int									root_FE_process;
		int 								FE_pcolor;

		int									start_timestep;
		double              				present_time;
		double								fe_timestep_length;
		int        							timestep;
		int        							newtonstep;

		ConditionalOStream 					dcout;

		parallel::shared::Triangulation<dim> triangulation;
		DoFHandler<dim>      				dof_handler;

		FESystem<dim>        				fe;
		const QGauss<dim>   				quadrature_formula;

		ConstraintMatrix     				hanging_node_constraints;
		std::vector<PointHistory<dim> > 	quadrature_point_history;

		PETScWrappers::MPI::SparseMatrix	system_matrix;
		PETScWrappers::MPI::SparseMatrix	mass_matrix;
//		PETScWrappers::MPI::SparseMatrix	system_inverse;
		PETScWrappers::MPI::Vector      	system_rhs;

		std::vector<types::global_dof_index> local_dofs_per_process;
		IndexSet 							locally_owned_dofs;
		IndexSet 							locally_relevant_dofs;
		unsigned int 						n_local_cells;

		double 								inc_vsupport;
		std::vector<bool> 					supp_boundary_dofs;
		std::vector<bool> 					clmp_boundary_dofs;
		std::vector<bool> 					load_boundary_dofs;

		double 								ll;
		double 								lls;
		double 								hh;
		double								hhs;
		double 								bb;
		double								diam_wght;

		std::vector<std::string> 			mdtype;
		Tensor<1,dim> 						cg_dir;

		std::string                         macrostatelocin;
		std::string                         macrostatelocout;
		std::string                         macrostatelocres;
		std::string                         macrologloc;

		int									freq_checkpoint;
		int									freq_output_visu;
		int									freq_output_lhist;

		bool 								activate_md_update;
	};



	template <int dim>
	FEProblem<dim>::FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for,
			std::vector<std::string> mdtype, Tensor<1,dim> cg_dir)
	:
		FE_communicator (dcomm),
		n_FE_processes (Utilities::MPI::n_mpi_processes(FE_communicator)),
		this_FE_process (Utilities::MPI::this_mpi_process(FE_communicator)),
		FE_pcolor (pcolor),
		dcout (std::cout,(this_FE_process == 0)),
		triangulation(FE_communicator),
		dof_handler (triangulation),
		fe (FE_Q<dim>(fe_deg), dim),
		quadrature_formula (quad_for),
		mdtype(mdtype),
		cg_dir(cg_dir)
	{}



	template <int dim>
	FEProblem<dim>::~FEProblem ()
	{
		dof_handler.clear ();
	}




	template <int dim>
	void FEProblem<dim>::make_grid ()
	{
		ll=0.150;
		lls=0.125;
		hh=0.100;
		hhs=0.075;
		bb=0.005;
		diam_wght=0.020;

		char filename[1024];
		sprintf(filename, "%s/mesh.tria", macrostatelocin.c_str());

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
			reps[0] = 45; reps[1] = 30; reps[2] = 2;
			GridGenerator::subdivided_hyper_rectangle(triangulation, reps, pp1, pp2);

			//triangulation.refine_global (1);

			// Saving triangulation, not usefull now and costly...
			/*sprintf(filename, "%s/mesh.tria", macrostatelocout.c_str());
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

		mass_matrix.reinit (locally_owned_dofs,
				locally_owned_dofs,
				sparsity_pattern,
				FE_communicator);
		system_matrix.reinit (locally_owned_dofs,
				locally_owned_dofs,
				sparsity_pattern,
				FE_communicator);
		system_rhs.reinit (locally_owned_dofs, FE_communicator);

		newton_update_displacement.reinit (dof_handler.n_dofs());
		incremental_displacement.reinit (dof_handler.n_dofs());
		displacement.reinit (dof_handler.n_dofs());
		old_displacement.reinit (dof_handler.n_dofs());
		for (unsigned int i=0; i<dof_handler.n_dofs(); ++i) old_displacement(i) = 0.0;

		newton_update_velocity.reinit (dof_handler.n_dofs());
		incremental_velocity.reinit (dof_handler.n_dofs());
		velocity.reinit (dof_handler.n_dofs());

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
	std::vector<Vector<double> > FEProblem<dim>::get_microstructure ()
	{
		// Generation of nanostructure based on size, weight ratio
		//generate_nanostructure();

		// Load flakes data (center position, angles, density)
		unsigned int npoints = 0;
		unsigned int nfchar = 0;
		std::vector<Vector<double> > structure_data (npoints, Vector<double>(nfchar));

		char filename[1024];
		sprintf(filename, "%s/structure_data.csv", macrostatelocin.c_str());

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
			dcout << "      Nboxes " << npoints << " - Nchar " << nfchar << std::endl;

			//dcout << "Char names: " << std::flush;
			if(getline(ifile, iline)){
				std::istringstream iss(iline);
				for(unsigned int k=0;k<nfchar;k++){
					getline(iss, ival, ',');
					//dcout << ival << " " << std::flush;
				}
			}
			//dcout << std::endl;

			structure_data.resize(npoints, Vector<double>(nfchar));
			for(unsigned int n=0;n<npoints;n++)
				if(getline(ifile, iline)){
					//dcout << "box: " << n << std::flush;
					std::istringstream iss(iline);
					for(unsigned int k=0;k<nfchar;k++){
						getline(iss, ival, ',');
						structure_data[n][k] = std::stof(ival);
						//dcout << " - " << structure_data[n][k] << std::flush;
					}
					//dcout << std::endl;
				}

			ifile.close();
		}
		else{
			dcout << "      Unable to open" << filename << " to read it, no microstructure loaded." << std::endl;
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
		mat = mdtype[0];

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
				for (int imat=1; imat<int(mdtype.size()); imat++)
					if(imat == int(structure_data[n][0])){
						mat = mdtype[imat];
					}

				//std::cout << " box number: " << n << " is in cell " << cell->active_cell_index()
				//  		  << " of material " << mat << std::endl;

				// Assembling the rotation matrix from the global orientation of the cell given by the
				// microstructure to the common ground direction
				rotam = compute_rotation_tensor(nglo, cg_dir);

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

		char filename[1024];

		// Set materials initial stiffness tensors
		std::vector<SymmetricTensor<4,dim> > stiffness_tensors (mdtype.size());
		std::vector<double > densities (mdtype.size());

		dcout << "    Importing initial stiffnesses and densities..." << std::endl;
		for(unsigned int imd=0;imd<mdtype.size();imd++){

			// Reading initial material stiffness tensor
			sprintf(filename, "%s/init.%s.stiff", macrostatelocout.c_str(), mdtype[imd].c_str());
			read_tensor<dim>(filename, stiffness_tensors[imd]);

			if(this_FE_process==0){
				std::cout << "       material: " << mdtype[imd].c_str() << "stiffness: " << std::endl;
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][0][0][0], stiffness_tensors[imd][0][0][1][1], stiffness_tensors[imd][0][0][2][2], stiffness_tensors[imd][0][0][0][1], stiffness_tensors[imd][0][0][0][2], stiffness_tensors[imd][0][0][1][2]);
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][1][1][0][0], stiffness_tensors[imd][1][1][1][1], stiffness_tensors[imd][1][1][2][2], stiffness_tensors[imd][1][1][0][1], stiffness_tensors[imd][1][1][0][2], stiffness_tensors[imd][1][1][1][2]);
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][2][2][0][0], stiffness_tensors[imd][2][2][1][1], stiffness_tensors[imd][2][2][2][2], stiffness_tensors[imd][2][2][0][1], stiffness_tensors[imd][2][2][0][2], stiffness_tensors[imd][2][2][1][2]);
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][1][0][0], stiffness_tensors[imd][0][1][1][1], stiffness_tensors[imd][0][1][2][2], stiffness_tensors[imd][0][1][0][1], stiffness_tensors[imd][0][1][0][2], stiffness_tensors[imd][0][1][1][2]);
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][0][2][0][0], stiffness_tensors[imd][0][2][1][1], stiffness_tensors[imd][0][2][2][2], stiffness_tensors[imd][0][2][0][1], stiffness_tensors[imd][0][2][0][2], stiffness_tensors[imd][0][2][1][2]);
				printf("           %+.4e %+.4e %+.4e %+.4e %+.4e %+.4e \n",stiffness_tensors[imd][1][2][0][0], stiffness_tensors[imd][1][2][1][1], stiffness_tensors[imd][1][2][2][2], stiffness_tensors[imd][1][2][0][1], stiffness_tensors[imd][1][2][0][2], stiffness_tensors[imd][1][2][1][2]);
			}

			sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), mdtype[imd].c_str());
				write_tensor<dim>(filename, stiffness_tensors[imd]);

			// Reading initial material density
			sprintf(filename, "%s/init.%s.density", macrostatelocout.c_str(), mdtype[imd].c_str());
				read_tensor<dim>(filename, densities[imd]);

			dcout << "       material: " << mdtype[imd].c_str() << "density: " << densities[imd] << std::endl;

			sprintf(filename, "%s/last.%s.density", macrostatelocout.c_str(), mdtype[imd].c_str());
				write_tensors<dim>(filename, densities[imd]);


		}

		// Setting up distributed quadrature point local history
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
		dcout << "    Loading microstructure..." << std::endl;
		std::vector<Vector<double> > structure_data;
		structure_data = get_microstructure();

		// Quadrature points data initialization and assigning material properties
		dcout << "    Assigning microstructure..." << std::endl;
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

					// Assign microstructure to the current cell (so far, mdtype
					// and rotation from global to common ground direction)
					if (q==0) assign_microstructure(cell, structure_data,
								local_quadrature_points_history[q].mat,
								local_quadrature_points_history[q].rotam);
					else{
						local_quadrature_points_history[q].mat = local_quadrature_points_history[0].mat;
						local_quadrature_points_history[q].rotam = local_quadrature_points_history[0].rotam;
					}

					// Apply stiffness and rotating it from the local sheet orientation (MD) to
					// global orientation (microstructure)
					for (int imd = 0; imd<int(mdtype.size()); imd++)
						if(local_quadrature_points_history[q].mat==mdtype[imd]){
							local_quadrature_points_history[q].new_stiff =
								rotate_tensor(stiffness_tensors[imd],
									transpose(local_quadrature_points_history[q].rotam));

							// Apply composite density (by averaging over replicas of given material)
							local_quadrature_points_history[q].rho = densities[imd];
						}
				}
			}
	}



	template <int dim>
	void FEProblem<dim>::restart ()
	{
		char filename[1024];

		// Recovery of the solution vector containing total displacements in the
		// previous simulation and computing the total strain from it.
		sprintf(filename, "%s/restart/lcts.solution.bin", macrostatelocin.c_str());
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
		else{
			dcout << "    No file to load/restart displacements from." << std::endl;
		}

		// Recovery of the velocity vector
		sprintf(filename, "%s/restart/lcts.velocity.bin", macrostatelocin.c_str());
		std::ifstream ifile_veloc(filename);
		if (ifile_veloc.is_open())
		{
			dcout << "    ...recovery of the velocity vector... " << std::flush;
			velocity.block_read(ifile_veloc);
			dcout << "    velocity norm: " << velocity.l2_norm() << std::endl;
			ifile_veloc.close();
		}
		else{
			dcout << "    No file to load/restart velocities from." << std::endl;
		}

		// Opening processor local history file
		sprintf(filename, "%s/restart/lcts.pr_%d.lhistory.bin", macrostatelocin.c_str(), this_FE_process);
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
					}
				}
			lhprocin.close();
		}
		else{
			dcout << "    No file to load/restart local histories from." << std::endl;
		}
	}


	// Might want to restructure this function to avoid repetitions
	// with boundary conditions correction performed at the end of the
	// assemble_system() function
	template <int dim>
	void FEProblem<dim>::set_boundary_values()
	{
		/*double tvel_vsupport_amplitude = 50.0; // target velocity of the boundary m/s-1

		double time_phase = 10.0*fe_timestep_length;

		int phase_no = std::floor(present_time/time_phase);

		dcout << present_time << " " << time_phase << " " << phase_no << std::endl;

		double tvel_vsupport = tvel_vsupport_amplitude;
		if (phase_no==0) tvel_vsupport*=1.0;
		else tvel_vsupport*=2.0;

		double acc_time=500.0*fe_timestep_length + fe_timestep_length*0.001; // duration during which the boundary accelerates s + slight delta for avoiding numerical error

		double acc_vsupport= tvel_vsupport/acc_time; // acceleration of the boundary m/s-2
		if (phase_no%2==0) acc_vsupport*=1.0;
		else if (phase_no%2==1) acc_vsupport*=-1.0;

		dcout << acc_vsupport << " " << tvel_vsupport << std::endl;*/

		double tacc_vsupport = 1.0e8; // acceleration of the boundary m/s-2

		double tvel_time=0.0*fe_timestep_length;
		double acc_time=10.0*fe_timestep_length + fe_timestep_length*0.001; // duration during which the boundary accelerates s + slight delta for avoiding numerical error

		bool is_loaded = true;

		dcout << "Loading condition: " << std::flush;
		// acceleration of the loading support (reaching aimed velocity)
		if (present_time<=acc_time){
			dcout << "ACCELERATE!!!" << std::flush;
			inc_vsupport = tacc_vsupport*fe_timestep_length;
		}
		// stationary motion of the loading support
		else if (present_time>acc_time and present_time<=acc_time+tvel_time){
			dcout << "CRUISING!!!" << std::flush;
			inc_vsupport = 0.0;
		}
		// deccelaration of the loading support (return to 0 velocity)
		else if (present_time>acc_time+tvel_time and present_time<=acc_time+tvel_time+acc_time){
			dcout << "DECCELERATE!!!" << std::flush;
			inc_vsupport = -1.0*tacc_vsupport*fe_timestep_length;
			is_loaded = false;
		}
		// stationary motion of the loading support
		else{
			dcout << "NOT LOADED!!!" << std::flush;
			inc_vsupport = 0.0;
			is_loaded = false;
		}

		dcout << " acceleration: " << tacc_vsupport << " - velocity increment: " << inc_vsupport << std::endl;

		FEValuesExtractors::Scalar x_component (dim-3);
		FEValuesExtractors::Scalar y_component (dim-2);
		FEValuesExtractors::Scalar z_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;

		supp_boundary_dofs.resize(dof_handler.n_dofs());
		clmp_boundary_dofs.resize(dof_handler.n_dofs());
		load_boundary_dofs.resize(dof_handler.n_dofs());

		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

		for ( ; cell != endc; ++cell) {

			double eps = (cell->minimum_vertex_distance());

			for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; ++face){
				unsigned int component;
				double value;

				for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v) {
					for (unsigned int c = 0; c < dim; ++c) {
						supp_boundary_dofs[cell->face(face)->vertex_dof_index (v, c)] = false;
						clmp_boundary_dofs[cell->face(face)->vertex_dof_index (v, c)] = false;
						load_boundary_dofs[cell->face(face)->vertex_dof_index (v, c)] = false;
					}

					double dcwght=sqrt((cell->face(face)->vertex(v)(0) - 0.)*(cell->face(face)->vertex(v)(0) - 0.)
							+ (cell->face(face)->vertex(v)(1) - 0.)*(cell->face(face)->vertex(v)(1) - 0.));

					if(is_loaded){
						if ((dcwght < diam_wght/2.) && (cell->face(face)->vertex(v)(2) - bb/2.) < eps/3.){
							value = -1.0*inc_vsupport;
							component = 2;
							load_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)] = true;
							boundary_values.insert(std::pair<types::global_dof_index, double>
							(cell->face(face)->vertex_dof_index (v, component), value));
						}
					}

					if ((fabs(cell->face(face)->vertex(v)(0)) > lls/2.
							|| fabs(cell->face(face)->vertex(v)(1)) > hhs/2.)
						  && ((cell->face(face)->vertex(v)(2) - -bb/2.) < eps/3.)){
						value = 0.;
						component = 2;
						supp_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->face(face)->vertex_dof_index (v, component), value));
					}

					/*if (fabs(cell->face(face)->vertex(v)(1) - 0.) < eps/3. && cell->face(face)->vertex(v)(0) < (ww - aa)){
						value = 0.;
						component = 1;
						clmp_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)] = true;
						boundary_values.insert(std::pair<types::global_dof_index, double>
						(cell->face(face)->vertex_dof_index (v, component), value));
					}*/
				}
			}
		}


		for (std::map<types::global_dof_index, double>::const_iterator
				p = boundary_values.begin();
				p != boundary_values.end(); ++p)
			incremental_velocity(p->first) = p->second;
	}



	template <int dim>
	double FEProblem<dim>::assemble_system (bool first_assemble)
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

		FullMatrix<double>   cell_mass (dofs_per_cell, dofs_per_cell);
		Vector<double>       cell_force (dofs_per_cell);

		FullMatrix<double>   cell_v_matrix (dofs_per_cell, dofs_per_cell);
		Vector<double>       cell_v_rhs (dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		BodyForce<dim>      body_force;
		std::vector<Vector<double> > body_force_values (n_q_points,
				Vector<double>(dim));

		system_rhs = 0;
		system_matrix = 0;

		for (; cell!=endc; ++cell)
			if (cell->is_locally_owned())
			{
				cell_mass = 0;
				cell_force = 0;

				cell_v_matrix = 0;
				cell_v_rhs = 0;

				fe_values.reinit (cell);

				const PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

				// Assembly of mass matrix
				if(first_assemble)
					for (unsigned int i=0; i<dofs_per_cell; ++i)
						for (unsigned int j=0; j<dofs_per_cell; ++j)
							for (unsigned int q_point=0; q_point<n_q_points;
									++q_point)
							{
								const double rho =
										local_quadrature_points_history[q_point].rho;

								const double
								phi_i = fe_values.shape_value (i,q_point),
								phi_j = fe_values.shape_value (j,q_point);

								// Non-zero value only if same dimension DOF, because
								// this is normally a scalar product of the shape functions vector
								int dcorr;
								if(i%dim==j%dim) dcorr = 1;
								else dcorr = 0;

								// Lumped mass matrix because the consistent one doesnt work...
								cell_mass(i,i) // cell_mass(i,j) instead...
								+= (rho * dcorr * phi_i * phi_j
										* fe_values.JxW (q_point));
							}

				// Assembly of external forces vector
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
						cell_force(i) += (
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

				// Assemble local matrices for v problem
				if(first_assemble) cell_v_matrix = cell_mass;

				//std::cout << "norm matrix " << cell_v_matrix.l1_norm() << " stiffness " << cell_stiffness.l1_norm() << std::endl;

				// Assemble local rhs for v problem
				cell_v_rhs.add(fe_timestep_length, cell_force);

				// Local to global for u and v problems
				if(first_assemble) hanging_node_constraints
										.distribute_local_to_global(cell_v_matrix, cell_v_rhs,
												local_dof_indices,
												system_matrix, system_rhs);
				else hanging_node_constraints
						.distribute_local_to_global(cell_v_rhs,
								local_dof_indices, system_rhs);
			}

		if(first_assemble){
			system_matrix.compress(VectorOperation::add);
			mass_matrix.copy_from(system_matrix);
		}
		else system_matrix.copy_from(mass_matrix);

		system_rhs.compress(VectorOperation::add);


		FEValuesExtractors::Scalar x_component (dim-3);
		FEValuesExtractors::Scalar y_component (dim-2);
		FEValuesExtractors::Scalar z_component (dim-1);
		std::map<types::global_dof_index,double> boundary_values;

		cell = dof_handler.begin_active(),
		endc = dof_handler.end();

		// Apply velocity boundary conditions
		double value = 0.;
		for ( ; cell != endc; ++cell) {
			for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; ++face){
				for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v) {
					for (unsigned int component = 0; component < dim; ++component) {

						if (load_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)])
						{
							boundary_values.insert(std::pair<types::global_dof_index, double>
							(cell->face(face)->vertex_dof_index (v, component), value));
						}

						if (supp_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)])
						{
							boundary_values.insert(std::pair<types::global_dof_index, double>
							(cell->face(face)->vertex_dof_index (v, component), value));
						}

						if (clmp_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)])
						{
							boundary_values.insert(std::pair<types::global_dof_index, double>
							(cell->face(face)->vertex_dof_index (v, component), value));
						}
					}
				}
			}
		}

		PETScWrappers::MPI::Vector tmp (locally_owned_dofs,FE_communicator);
		MatrixTools::apply_boundary_values (boundary_values,
				system_matrix,
				tmp,
				system_rhs,
				false);
		newton_update_velocity = tmp;

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
		distributed_newton_update = newton_update_velocity;

		// The residual used internally to test solver convergence is
		// not identical to ours, it probably considers preconditionning.
		// Therefore, extra precision is required in the solver proportionnaly
		// to the norm of the system matrix, to reduce sufficiently our residual
		SolverControl       solver_control (dof_handler.n_dofs(),
				1e-03);

		PETScWrappers::SolverCG cg (solver_control,
				FE_communicator);

		// Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
		// not optimal for large scale simulations.
		PETScWrappers::PreconditionJacobi preconditioner(system_matrix);
		cg.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_velocity = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_velocity);

		dcout << "    FE Solver - norm of newton update is " << newton_update_velocity.l2_norm()
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
		distributed_newton_update = newton_update_velocity;

		// The residual used internally to test solver convergence is
		// not identical to ours, it probably considers preconditionning.
		// Therefore, extra precision is required in the solver proportionnaly
		// to the norm of the system matrix, to reduce sufficiently our residual
		SolverControl       solver_control (dof_handler.n_dofs(),
				1e-03);

		PETScWrappers::SolverGMRES gmres (solver_control,
				FE_communicator);

		// Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
		// not optimal for large scale simulations.
		PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
		gmres.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_velocity = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_velocity);

		dcout << "    FE Solver - norm of newton update is " << newton_update_velocity.l2_norm()
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
		distributed_newton_update = newton_update_velocity;

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
				1e-03);

		PETScWrappers::SolverBicgstab bicgs (solver_control,
				FE_communicator);

		bicgs.solve (system_matrix, distributed_newton_update, system_rhs,
				preconditioner);

		newton_update_velocity = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_velocity);

		dcout << "    FE Solver - norm of newton update is " << newton_update_velocity.l2_norm()
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
		distributed_newton_update = newton_update_velocity;

		SolverControl       solver_control;

		PETScWrappers::SparseDirectMUMPS solver (solver_control,
				FE_communicator);

		//solver.set_symmetric_mode(false);

		solver.solve (system_matrix, distributed_newton_update, system_rhs);
		//system_inverse.vmult(distributed_newton_update, system_rhs);

		newton_update_velocity = distributed_newton_update;
		hanging_node_constraints.distribute (newton_update_velocity);

		dcout << "    FE Solver - norm of newton update is " << newton_update_velocity.l2_norm()
							  << std::endl;
	}



	template <int dim>
	void FEProblem<dim>::update_incremental_variables ()
	{
		// Displacement newton update is equal to the current velocity multiplied by the timestep length
		newton_update_displacement.equ(fe_timestep_length, velocity);
		newton_update_displacement.add(fe_timestep_length, incremental_velocity);
		newton_update_displacement.add(fe_timestep_length, newton_update_velocity);
		newton_update_displacement.add(-1.0, incremental_displacement);

		//hcout << "    Upd. Norms: " << fe_problem.newton_update_displacement.l2_norm() << " - " << fe_problem.newton_update_velocity.l2_norm() <<  std::endl;

		//fe_problem.newton_update_displacement.equ(fe_timestep_length, fe_problem.newton_update_velocity);

		incremental_velocity.add (1.0, newton_update_velocity);
		incremental_displacement.add (1.0, newton_update_displacement);
		//hcout << "    Inc. Norms: " << fe_problem.incremental_displacement.l2_norm() << " - " << fe_problem.incremental_velocity.l2_norm() <<  std::endl;
	}




	template <int dim>
	void FEProblem<dim>::update_strain_quadrature_point_history(const Vector<double>& displacement_update)
	{
		// Create file with qptid to update at timeid
		std::ofstream ofile;
		char update_local_filename[1024];
		sprintf(update_local_filename, "%s/last.%d.qpupdates", macrostatelocout.c_str(), this_FE_process);
		ofile.open (update_local_filename);

		// Create file with mdtype of qptid to update at timeid
		std::ofstream omatfile;
		char mat_update_local_filename[1024];
		sprintf(mat_update_local_filename, "%s/last.%d.matqpupdates", macrostatelocout.c_str(), this_FE_process);
		omatfile.open (mat_update_local_filename);

		// Preparing requirements for strain update
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		char time_id[1024]; sprintf(time_id, "%d-%d", timestep, newtonstep);

		if (newtonstep > 0) dcout << "        " << "...checking quadrature points requiring update..." << std::endl;

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				SymmetricTensor<2,dim> newton_strain_tensor, avg_upd_strain_tensor;
				SymmetricTensor<2,dim> avg_new_strain_tensor, avg_new_stress_tensor;

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

				avg_upd_strain_tensor = 0.;
				avg_new_strain_tensor = 0.;
				avg_new_stress_tensor = 0.;

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					local_quadrature_points_history[q].old_strain =
							local_quadrature_points_history[q].new_strain;

					local_quadrature_points_history[q].old_stress =
							local_quadrature_points_history[q].new_stress;

					local_quadrature_points_history[q].old_stiff =
							local_quadrature_points_history[q].new_stiff;

					if (newtonstep == 0) local_quadrature_points_history[q].inc_strain = 0.;

					// Strain tensor update
					local_quadrature_points_history[q].newton_strain = get_strain (displacement_update_grads[q]);
					local_quadrature_points_history[q].inc_strain += local_quadrature_points_history[q].newton_strain;
					local_quadrature_points_history[q].new_strain += local_quadrature_points_history[q].newton_strain;
					local_quadrature_points_history[q].upd_strain += local_quadrature_points_history[q].newton_strain;

					for(unsigned int k=0;k<dim;k++)
						for(unsigned int l=k;l<dim;l++){
							avg_upd_strain_tensor[k][l] += local_quadrature_points_history[q].upd_strain[k][l];
							avg_new_strain_tensor[k][l] += local_quadrature_points_history[q].new_strain[k][l];
							avg_new_stress_tensor[k][l] += local_quadrature_points_history[q].new_stress[k][l];
						}
				}

				for(unsigned int k=0;k<dim;k++)
					for(unsigned int l=k;l<dim;l++){
						avg_upd_strain_tensor[k][l] /= quadrature_formula.size();
						avg_new_strain_tensor[k][l] /= quadrature_formula.size();
						avg_new_stress_tensor[k][l] /= quadrature_formula.size();
					}

				// Uncomment one on the 4 following "if" statement to derive stress tensor from MD for:
				//   (i) all cells,
				//  (ii) cells in given location,
				// (iii) cells based on their id,
				//  (iv) none of the cells
				if (activate_md_update)
				//if (cell->barycenter()(1) <  3.0*tt && cell->barycenter()(0) <  1.10*(ww - aa) && cell->barycenter()(0) > 0.0*(ww - aa))
				/*if ((cell->active_cell_index() == 2922 || cell->active_cell_index() == 2923
					|| cell->active_cell_index() == 2924 || cell->active_cell_index() == 2487
					|| cell->active_cell_index() == 2488 || cell->active_cell_index() == 2489))*/ // For debug...
				{
					for (unsigned int qc=0; qc<quadrature_formula.size(); ++qc)
						local_quadrature_points_history[qc].to_be_updated = true;

					// The cell will get its stress from MD, but should it run an MD simulation?
					if (true
						// otherwise MD simulation unecessary, because no significant volume change and MD will fail
						/*avg_upd_strain_tensor.norm() > 1.0e-7*/
						// in case of extreme straining with reaxff
						/*&& (avg_new_stress_tensor.norm() > 1.0e8 || avg_new_strain_tensor.norm() < 3.0)*/
						){
						std::cout << "           "
								<< " cell "<< cell->active_cell_index()
								<< " upd norm " << avg_upd_strain_tensor.norm()
								<< " total norm " << avg_new_strain_tensor.norm()
								<< " total stress norm " << avg_new_stress_tensor.norm()
								<< std::endl;

						// Write strains since last update in a file named ./macrostate_storage/last.cellid-qid.strain
						char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());
						char filename[1024];

						SymmetricTensor<2,dim> rot_avg_upd_strain_tensor;

						rot_avg_upd_strain_tensor =
									rotate_tensor(avg_upd_strain_tensor, local_quadrature_points_history[0].rotam);

						sprintf(filename, "%s/last.%s.upstrain", macrostatelocout.c_str(), cell_id);
						write_tensor<dim>(filename, rot_avg_upd_strain_tensor);

						ofile << cell_id << std::endl;
						omatfile << local_quadrature_points_history[0].mat << std::endl;
					}
				}
				else{
					for (unsigned int qc=0; qc<quadrature_formula.size(); ++qc)
						local_quadrature_points_history[qc].to_be_updated = false;
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

			sprintf(update_filename, "%s/last.qpupdates", macrostatelocout.c_str());
			outfile.open (update_filename);
			for (int ip=0; ip<n_FE_processes; ip++){
				sprintf(update_local_filename, "%s/last.%d.qpupdates", macrostatelocout.c_str(), ip);
				infile.open (update_local_filename);
				while (getline(infile, iline)) outfile << iline << std::endl;
				infile.close();
			}
			outfile.close();

                        char alltime_update_filename[1024];
                        sprintf(alltime_update_filename, "%s/alltime_cellupdates.dat", macrologloc.c_str());
                        outfile.open (alltime_update_filename, std::ofstream::app);
                        if(timestep==start_timestep && newtonstep==1) outfile << "timestep,newtonstep,cell" << std::endl;
                        infile.open (update_filename);
                        while (getline(infile, iline)) outfile << timestep << "," << newtonstep << "," << iline << std::endl;
                        infile.close();
                        outfile.close();

			sprintf(update_filename, "%s/last.matqpupdates", macrostatelocout.c_str());
			outfile.open (update_filename);
			for (int ip=0; ip<n_FE_processes; ip++){
				sprintf(update_local_filename, "%s/last.%d.matqpupdates", macrostatelocout.c_str(), ip);
				infile.open (update_local_filename);
				while (getline(infile, iline)) outfile << iline << std::endl;
				infile.close();
			}
			outfile.close();
		}
	}




	template <int dim>
	void FEProblem<dim>::update_stress_quadrature_point_history(const Vector<double>& displacement_update)
	{
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		char time_id[1024]; sprintf(time_id, "%d-%d", timestep, newtonstep);

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

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					if (newtonstep == 0) local_quadrature_points_history[q].inc_stress = 0.;

					if (local_quadrature_points_history[q].to_be_updated){

						// Updating stiffness tensor
						/*SymmetricTensor<4,dim> stmp_stiff;
						sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), cell_id);
						read_tensor<dim>(filename, stmp_stiff);

						// Rotate the output stiffness wrt the flake angles
						local_quadrature_points_history[q].new_stiff =
								rotate_tensor(stmp_stiff, transpose(local_quadrature_points_history[q].rotam));
						 */

						// Updating stress tensor
						bool load_stress;

						/*SymmetricTensor<4,dim> loc_stiffness;
						sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), cell_id);
						read_tensor<dim>(filename, loc_stiffness);*/

						SymmetricTensor<2,dim> loc_stress;
						sprintf(filename, "%s/last.%s.stress", macrostatelocout.c_str(), cell_id);
						load_stress = read_tensor<dim>(filename, loc_stress);

						// Rotate the output stress wrt the flake angles
						if (load_stress) local_quadrature_points_history[q].new_stress =
									rotate_tensor(loc_stress, transpose(local_quadrature_points_history[q].rotam));
						else local_quadrature_points_history[q].new_stress +=
                                                        0.00*local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].newton_strain;

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
					/*sprintf(filename, "%s/last.%s-%d.stress", macrostatelocout.c_str(), cell_id,q);
					write_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);*/

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



	template <int dim>
	void FEProblem<dim>::clean_transfer()
	{
		char filename[1024];

		// Removing lists of material types of quadrature points to update per procs
		sprintf(filename, "%s/last.%d.matqpupdates", macrostatelocout.c_str(), this_FE_process);
		remove(filename);

		// Removing lists of quadrature points to update per procs
		sprintf(filename, "%s/last.%d.qpupdates", macrostatelocout.c_str(), this_FE_process);
		remove(filename);

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				char cell_id[1024]; sprintf(cell_id, "%d", cell->active_cell_index());

				// Removing stiffness passing file
				//sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), cell_id);
				//remove(filename);

				// Removing stress passing file
				sprintf(filename, "%s/last.%s.stress", macrostatelocout.c_str(), cell_id);
				remove(filename);

				// Removing updstrain passing file
				sprintf(filename, "%s/last.%s.upstrain", macrostatelocout.c_str(), cell_id);
				remove(filename);
			}
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
	void FEProblem<dim>::output_lhistory ()
	{
		char filename[1024];

		// Initialization of the processor local history data file
		sprintf(filename, "%s/pr_%d.lhistory.csv", macrologloc.c_str(), this_FE_process);
		std::ofstream  lhprocout(filename, std::ios_base::app);
		long cursor_position = lhprocout.tellp();

		if (cursor_position == 0)
		{
			lhprocout << "timestep,time,cell,qpoint,material";
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
					lhprocout << timestep
							<< present_time
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
		void FEProblem<dim>::output_visualisation ()
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

			// Output of velocity as a vector
			std::vector<std::string>  velocity_names (dim, "velocity");
			data_out.add_data_vector (velocity,
					velocity_names,
					DataOut<dim>::type_dof_data,
					data_component_interpretation);

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


			// Output of the cell id
			Vector<double> cell_ids (triangulation.n_active_cells());
			{
				typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(),
				endc = triangulation.end();
				for (; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						cell_ids(cell->active_cell_index())
								= cell->active_cell_index();
					}
					else cell_ids(cell->active_cell_index()) = -1;
			}
			data_out.add_data_vector (cell_ids, "cellID");


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
			std::string filename = smacrologloc + "/" + "solution-" + Utilities::int_to_string(timestep,4)
			+ "." + Utilities::int_to_string(this_FE_process,3)
			+ ".vtu";
			AssertThrow (n_FE_processes < 1000, ExcNotImplemented());

			std::ofstream output (filename.c_str());
			data_out.write_vtu (output);

			if (this_FE_process==0)
			{
				std::vector<std::string> filenames_loc;
				for (int i=0; i<n_FE_processes; ++i)
					filenames_loc.push_back ("solution-" + Utilities::int_to_string(timestep,4)
				+ "." + Utilities::int_to_string(i,3)
				+ ".vtu");

				const std::string
				visit_master_filename = (smacrologloc + "/" + "solution-" +
						Utilities::int_to_string(timestep,4) +
						".visit");
				std::ofstream visit_master (visit_master_filename.c_str());
				//data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
				DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

				const std::string
				pvtu_master_filename = (smacrologloc + "/" + "solution-" +
						Utilities::int_to_string(timestep,4) +
						".pvtu");
				std::ofstream pvtu_master (pvtu_master_filename.c_str());
				data_out.write_pvtu_record (pvtu_master, filenames_loc);

				static std::vector<std::pair<double,std::string> > times_and_names;
				const std::string
							pvtu_master_filename_loc = ("solution-" +
									Utilities::int_to_string(timestep,4) +
									".pvtu");
				times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename_loc));
				std::ofstream pvd_output (smacrologloc + "/" + "solution.pvd");
				//data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
				DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
			}
		}



	template <int dim>
	void FEProblem<dim>::output_results ()
	{
		// Output local history by processor
		if(timestep%freq_output_lhist==0) output_lhistory ();

		// Output visualisation files for paraview
		if(timestep%freq_output_visu==0) output_visualisation();
	}



	// Creation of a checkpoint with the bare minimum data to restart the simulation (i.e nodes information,
	// and quadrature point information)
	template <int dim>
	void FEProblem<dim>::checkpoint (char* timeid) const
	{
		char filename[1024];

		// Copy of the solution vector at the end of the presently converged time-step.
		if (this_FE_process==0)
		{
			// Write solution vector to binary for simulation restart
			const std::string solution_filename = (macrostatelocres + "/" + timeid + ".solution.bin");
			std::ofstream ofile(solution_filename);
			displacement.block_write(ofile);
			ofile.close();

			const std::string solution_filename_veloc = (macrostatelocres + "/" + timeid + ".velocity.bin");
			std::ofstream ofile_veloc(solution_filename_veloc);
			velocity.block_write(ofile_veloc);
			ofile_veloc.close();
		}


		// Output of the last converged timestep quadrature local history per processor
		sprintf(filename, "%s/%s.pr_%d.lhistory.bin", macrostatelocres.c_str(), timeid, this_FE_process);
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
		MPI_Barrier(FE_communicator);
	}



	template <int dim>
	void FEProblem<dim>::init (int sstp, double tlength,
							   std::string mslocin, std::string mslocout,
							   std::string mslocres, std::string mlogloc,
							   int fchpt, int fovis, int folhis, bool actmdup,
							   std::vector<std::string> mdt, Tensor<1,dim> cgd){

		// Setting up checkpoint and output frequencies
		freq_checkpoint = fchpt;
		freq_output_visu = fovis;
		freq_output_lhist = folhis;

		// Setting up usage of MD to update constitutive behaviour
		activate_md_update = actmdup;

		// Setting up starting timestep number and timestep length
		start_timestep = sstp;
		fe_timestep_length = tlength;

		// Setting up directories location
		macrostatelocin = mslocin;
		macrostatelocout = mslocout;
		macrostatelocres = mslocres;
		macrologloc = mlogloc;

		// Setting materials name list
		mdtype = mdt;

		// Setting up common ground direction for rotation from microstructure given orientation
		cg_dir = cgd;

		dcout << " Initiation of the Mesh...       " << std::endl;
		make_grid ();

		dcout << " Initiation of the global vectors and tensor...       " << std::endl;
		setup_system ();

		dcout << " Initiation of the local tensors...       " << std::endl;
		setup_quadrature_point_history ();
		MPI_Barrier(FE_communicator);

		dcout << " Loading previous simulation data...       " << std::endl;
		restart ();
	}



	template <int dim>
	void FEProblem<dim>::beginstep(int tstp, double ptime){
		timestep = tstp;
		present_time = ptime;

		incremental_velocity = 0;
		incremental_displacement = 0;

		// Setting boudary conditions for current timestep
		set_boundary_values ();

	}



	template <int dim>
	void FEProblem<dim>::solve (int nstp){

		newtonstep = nstp;

		double previous_res;

		dcout << "    Initial assembling FE system..." << std::flush;
		if(timestep==start_timestep) previous_res = assemble_system (true);
		else previous_res = assemble_system (false);

		dcout << "    Initial residual: "
				<< previous_res
				<< std::endl;

		dcout << "    Beginning of timestep: " << timestep << " - newton step: " << newtonstep << std::flush;
		dcout << "    Solving FE system..." << std::flush;

		// Solving for the update of the increment of velocity
		solve_linear_problem_CG();

		// Updating incremental variables
		update_incremental_variables();

		MPI_Barrier(FE_communicator);
		dcout << "    Updating quadrature point data..." << std::endl;

		update_strain_quadrature_point_history(newton_update_displacement);

	}



	template <int dim>
	bool FEProblem<dim>::check (){
		double previous_res;

		update_stress_quadrature_point_history (newton_update_displacement);

		dcout << "    Re-assembling FE system..." << std::flush;
		previous_res = assemble_system (false);
		MPI_Barrier(FE_communicator);

		// Cleaning temporary files (nanoscale logs and FE/MD data transfer)
		clean_transfer();

		MPI_Barrier(FE_communicator);

		dcout << "    Residual: "
				<< previous_res
				<< std::endl;

		bool continue_newton = false;
		//if (previous_res>1e-02 and newtonstep < 5) continue_newton = true;

		return continue_newton;

	}



	template <int dim>
	void FEProblem<dim>::endstep (){

		// Updating the total displacement and velocity vectors
		velocity+=incremental_velocity;
		displacement+=incremental_displacement;
		old_displacement=displacement;

		// Outputs
		output_results ();

		// Saving files for restart
		if(timestep%freq_restart==0){
			char timeid[1024];
			sprintf(timeid, "%s", "lcts");
			checkpoint (timeid);
			sprintf(timeid, "%d", timestep);
			checkpoint (timeid);
		}

		dcout << std::endl;
	}





	template <int dim>
	class MMDProblem
	{
	public:
		MMDProblem (MPI_Comm mcomm, int pcolor);
		~MMDProblem ();
		void init_mmd (int sstp, double mdtlength, double mdtemp, int nss, double strr,
				   std::string nslocin, std::string nslocout, std::string nslocres, std::string nlogloc,
				   std::string nlogloctmp,std::string nloglochom, std::string mslocout, std::string mdsdir,
				   int fchpt, int fohom, unsigned int bnmin, unsigned int mppn,
				   std::vector<std::string> mdt, Tensor<1,dim> cgd, unsigned int nr, bool ups);
		void update_mmd (int tstp, double ptime, int nstp);

	private:
		void restart ();
		void setup_replica_data();
		void initialize_replicas ();

		void set_md_procs (int nmdruns);

		// stored in run_single_md.bak
		//void lammps_homogenization ();
		//void lammps_straining ();
		//void run_single_md(char* ctime, char* ccell, const char* cmat, unsigned int repl, std::string qpreplogloc);

		void prepare_md_simulations();

		void execute_inside_md_simulations();

		void write_proc_job_list_json(list_proc_jobs_json, time_id, max_nodes_per_md);
		void concatenate_job_list(list_jobs_json);
		void execute_pjm_md_simulations();

		void store_md_simulations();

		MPI_Comm 							mmd_communicator;
		MPI_Comm 							md_batch_communicator;
		int 								mmd_n_processes;
		int 								md_batch_n_processes;
		int 								n_md_batches;
		int 								this_mmd_process;
		int 								this_md_batch_process;
		int 								mmd_pcolor;
		int									md_batch_pcolor;

		unsigned int						ncupd;

		unsigned int						machine_ppn;
		unsigned int						batch_nnodes_min;

		ConditionalOStream 					mcout;

		int									start_timestep;
		double              				present_time;
		int        							timestep;
		int        							newtonstep;

		std::string 						time_id;
		std::vector<std::string>			cell_id;
		std::vector<std::string>			cell_mat;
		std::vector<std::string>			qpreplogloc;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;
		std::vector<ReplicaData<dim> > 		replica_data;
		Tensor<1,dim> 						cg_dir;

		double								md_timestep_length;
		double								md_temperature;
		int									md_nsteps_sample;
		double								md_strain_rate;

		int									freq_checkpoint;
		int									freq_output_homog;

		bool 								output_homog;
		bool 								checkpoint_save;

		std::string                         macrostatelocout;

		std::string                         nanostatelocin;
		std::string							nanostatelocout;
		std::string							nanostatelocres;
		std::string							nanologloc;
		std::string							nanologloctmp;
		std::string							nanologlochom;

		std::string							md_scripts_directory;
		bool								use_pjm_scheduler;

	};



	template <int dim>
	MMDProblem<dim>::MMDProblem (MPI_Comm mcomm, int pcolor)
	:
		mmd_communicator (mcomm),
		mmd_n_processes (Utilities::MPI::n_mpi_processes(mmd_communicator)),
		this_mmd_process (Utilities::MPI::this_mpi_process(mmd_communicator)),
		mmd_pcolor (pcolor),
		mcout (std::cout,(this_mmd_process == 0))
	{}



	template <int dim>
	MMDProblem<dim>::~MMDProblem ()
	{}




	template <int dim>
	void MMDProblem<dim>::restart ()
	{
		// Cleaning the log files for all the MD simulations of the current timestep
		if (this_md_process==0)
		{
			char command[1024];
			// Clean "nanoscale_logs" of the finished timestep
			sprintf(command, "for ii in `ls %s/restart/ | grep -o '[^-]*$' | cut -d. -f2-`; "
							 "do cp %s/restart/lcts.${ii} %s/last.${ii}; "
							 "done", nanostatelocin.c_str(), nanostatelocin.c_str(), nanostatelocout.c_str());
			int ret = system(command);
			if (ret!=0){
				std::cerr << "Failed to copy input restart files (lcts) of the MD simulations as current output (last)!" << std::endl;
				exit(1);
			}
		}
	}




	template <int dim>
	void MMDProblem<dim>::setup_replica_data ()
	{
	    using boost::property_tree::ptree;

	    char filename[1024];
		for(unsigned int imd=0; imd<mdtype.size(); imd++)
			for(unsigned int irep=0; irep<nrepl; irep++){
				sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(), mdtype[imd].c_str(), irep+1);
				if(!file_exists(filename)){
					std::cerr << "Missing data for replica #" << irep+1
							  << " of material" << mdtype[imd].c_str()
							  << "." << std::endl;
					exit(1);
				}
			}

		replica_data.resize(nrepl * mdtype.size());
		for(unsigned int imd=0; imd<mdtype.size(); imd++)
			for(unsigned int irep=0; irep<nrepl; irep++){
				// Setting material name and replica number
				replica_data[imd*nrepl+irep].mat=mdtype[imd];
				replica_data[imd*nrepl+irep].repl=irep+1;

				// Initializing mechanical characteristics after equilibration
				replica_data[imd*nrepl+irep].length = 0;
				replica_data[imd*nrepl+irep].init_stress = 0;
				replica_data[imd*nrepl+irep].init_stiffness = 0;

				// Parse JSON data file
			    sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(),
			    		replica_data[imd*nrepl+irep].mat.c_str(), replica_data[imd*nrepl+irep].repl);

			    std::ifstream jsonFile(filename);
			    ptree pt;
			    try{
				    read_json(jsonFile, pt);
			    }
			    catch (const boost::property_tree::json_parser::json_parser_error& e)
			    {
			        hcout << "Invalid JSON replica data input file (" << filename << ")" << std::endl;  // Never gets here
			    }

			    // Printing the whole tree of the JSON file
			    //bptree_print(pt);

				// Load density of given replica of given material
			    std::string rdensity = bptree_read(pt, "relative_density");
				replica_data[imd*nrepl+irep].rho = std::stod(rdensity)*1000.;

				// Load number of flakes in box
			    std::string numflakes = bptree_read(pt, "Nsheets");
				replica_data[imd*nrepl+irep].nflakes = std::stoi(numflakes);

				/*hcout << "Hi repl: " << replica_data[imd*nrepl+irep].repl
					  << " - mat: " << replica_data[imd*nrepl+irep].mat
					  << " - rho: " << replica_data[imd*nrepl+irep].rho
					  << std::endl;*/

				// Load replica orientation (normal to flake plane if composite)
				if(replica_data[imd*nrepl+irep].nflakes==1){
					std::string fvcoorx = bptree_read(pt, "normal_vector","1","x");
					std::string fvcoory = bptree_read(pt, "normal_vector","1","y");
					std::string fvcoorz = bptree_read(pt, "normal_vector","1","z");
					//hcout << fvcoorx << " " << fvcoory << " " << fvcoorz <<std::endl;
					Tensor<1,dim> nvrep;
					nvrep[0]=std::stod(fvcoorx);
					nvrep[1]=std::stod(fvcoory);
					nvrep[2]=std::stod(fvcoorz);
					// Set the rotation matrix from the replica orientation to common
					// ground FE/MD orientation (arbitrary choose x-direction)
					replica_data[imd*nrepl+irep].rotam=compute_rotation_tensor(nvrep,cg_dir);
				}
				else{
					Tensor<2,dim> idmat;
					idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;
					// Simply fill the rotation matrix with the identity matrix
					replica_data[imd*nrepl+irep].rotam=idmat;
				}
			}
	}





	template <int dim>
	void MMDProblem<dim>::initialize_replicas ()
	{
		// Number of MD simulations at this iteration...
		int nmdruns = mdtype.size()*nrepl;

		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		set_md_procs(nmdruns);

		for(unsigned int imdt=0;imdt<mdtype.size();imdt++)
		{
			// type of MD box (so far PE or PNC)
			std::string mdt = mdtype[imdt];

			for(unsigned int repl=0;repl<nrepl;repl++)
			{
				int imdrun=imdt*nrepl + (repl);
				if (lammps_pcolor == (imdrun%n_md_batches))
				{
					// Offset replica number because in filenames, replicas start at 1
					int numrepl = repl+1;

					std::vector<double> 				initial_length (dim);
					SymmetricTensor<2,dim> 				initial_stress_tensor;
					SymmetricTensor<4,dim> 				initial_stiffness_tensor;

					char macrofilenamein[1024];
					sprintf(macrofilenamein, "%s/init.%s_%d.stiff", macrostatelocin.c_str(), mdt.c_str(), numrepl);
					char macrofilenameout[1024];
					sprintf(macrofilenameout, "%s/init.%s_%d.stiff", macrostatelocout.c_str(), mdt.c_str(), numrepl);
					bool macrostate_exists = file_exists(macrofilenamein);

					char macrofilenameinstress[1024];
					sprintf(macrofilenameinstress, "%s/init.%s_%d.stress", macrostatelocin.c_str(), mdt.c_str(), numrepl);
					char macrofilenameoutstress[1024];
					sprintf(macrofilenameoutstress, "%s/init.%s_%d.stress", macrostatelocout.c_str(), mdt.c_str(), numrepl);
					bool macrostatestress_exists = file_exists(macrofilenameinstress);

					char macrofilenameinlength[1024];
					sprintf(macrofilenameinlength, "%s/init.%s_%d.length", macrostatelocin.c_str(), mdt.c_str(), numrepl);
					char macrofilenameoutlength[1024];
					sprintf(macrofilenameoutlength, "%s/init.%s_%d.length", macrostatelocout.c_str(), mdt.c_str(), numrepl);
					bool macrostatelength_exists = file_exists(macrofilenameinlength);

					char nanofilenamein[1024];
					sprintf(nanofilenamein, "%s/init.%s_%d.bin", nanostatelocin.c_str(), mdt.c_str(), numrepl);
					char nanofilenameout[1024];
					sprintf(nanofilenameout, "%s/init.%s_%d.bin", nanostatelocout.c_str(), mdt.c_str(), numrepl);
					bool nanostate_exists = file_exists(nanofilenamein);

					if(!macrostate_exists || !macrostatestress_exists || !macrostatelength_exists || !nanostate_exists){
						if(this_md_batch_process == 0)
							std::cerr << "Missing equilibrated initial data for material " << mdt.c_str() << " replica #" << numrepl << " ("
									  << "stiffness file: " << macrostate_exists
									  << " stress file: " << macrostatestress_exists
									  << " dimensions file: " << macrostatelength_exists
									  << " binary state file: " << nanostate_exists
									  << ")"<< std::endl;
						MPI_Abort(world_communicator, 1);
					}
					else{
						if(this_md_batch_process == 0){
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

		if(this_md_batch_process == 0){
			for(unsigned int imd=0;imd<mdtype.size();imd++)
			{
				SymmetricTensor<4,dim> initial_stiffness_tensor;
				initial_stiffness_tensor = 0.;

				double initial_density = 0.;

				// type of MD box (so far PE or PNC)
				std::string mdt = mdtype[imd];

				for(unsigned int repl=0;repl<nrepl;repl++)
				{
					char macrofilenamein[1024];
					sprintf(macrofilenamein, "%s/init.%s_%d.stiff", macrostatelocout.c_str(), mdt.c_str(), repl+1);

					SymmetricTensor<4,dim> 				initial_rep_stiffness_tensor;
					SymmetricTensor<4,dim> 				cg_initial_rep_stiffness_tensor;

					read_tensor<dim>(macrofilenamein, initial_rep_stiffness_tensor);

					// Rotate tensor from replica orientation to common ground
					cg_initial_rep_stiffness_tensor =
							rotate_tensor(initial_rep_stiffness_tensor, replica_data[imd*nrepl+repl].rotam);

					// Averaging tensors in the common ground referential
					initial_stiffness_tensor += cg_initial_rep_stiffness_tensor;

					// Debugs...
					/*SymmetricTensor<4,dim> 				orig_cg_initial_rep_stiffness_tensor;
					orig_cg_initial_rep_stiffness_tensor =
												rotate_tensor(cg_initial_rep_stiffness_tensor, transpose(replica_data[imd*nrepl+repl].rotam));
					char macrofilenameout[1024];
					sprintf(macrofilenameout, "%s/init.%s_%d.stiff_cg", macrostatelocout.c_str(), mdt.c_str(), repl+1);
					write_tensor<dim>(macrofilenameout, cg_initial_rep_stiffness_tensor);
					sprintf(macrofilenameout, "%s/init.%s_%d.stiff_cg_orig", macrostatelocout.c_str(), mdt.c_str(), repl+1);
					write_tensor<dim>(macrofilenameout, orig_cg_initial_rep_stiffness_tensor);*/

					// Averaging density over replicas
					initial_density += replica_data[imd*nrepl+repl].rho;
				}

				initial_stiffness_tensor /= nrepl;
				initial_density /= nrepl;

				char macrofilenameout[1024];
				sprintf(macrofilenameout, "%s/init.%s.stiff", macrostatelocout.c_str(), mdt.c_str());
				write_tensor<dim>(macrofilenameout, initial_stiffness_tensor);

				char macrofilenameout[1024];
				sprintf(macrofilenameout, "%s/init.%s.density", macrostatelocout.c_str(), mdt.c_str());
				write_tensor<dim>(macrofilenameout, initial_density);
			}
		}
	}




	// There are several number of processes encountered: (i) n_lammps_processes the highest provided
	// as an argument to aprun, (ii) ND the number of processes provided to deal.ii
	// [arbitrary], (iii) NI the number of processes provided to the lammps initiation
	// [as close as possible to n_world_processes], and (iv) n_lammps_processes_per_batch the number of processes provided to one lammps
	// testing [NT divided by n_lammps_batch the number of concurrent testing boxes].
	template <int dim>
	void MMDProblem<dim>::set_md_procs (int nmdruns)
	{
		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		int npbtch_min = batch_nnodes_min*machine_ppn;
		//int nbtch_max = int(n_world_processes/npbtch_min);

		//int nrounds = int(nmdruns/nbtch_max)+1;
		//int nmdruns_round = nmdruns/nrounds;

		int fair_npbtch = int(mmd_n_processes/(nmdruns));

		int npb = std::max(npbtch_min, fair_npbtch - fair_npbtch%npbtch_min);
		//int nbtch = int(n_world_processes/npbtch);

		// Arbitrary setting of NB and NT
		md_batch_n_processes = npb;

		n_md_batches = int(mmd_n_processes/md_batch_n_processes);
		if(n_md_batches == 0) {n_md_batches=1; md_batch_n_processes=mmd_n_processes;}

		mcout << "        " << "...number of processes per batches: " << md_batch_n_processes
							<< "   ...number of batches: " << n_md_batches << std::endl;

		md_batch_pcolor = MPI_UNDEFINED;

		// LAMMPS processes color: regroup processes by batches of size NB, except
		// the last ones (me >= NB*NC) to create batches of only NB processes, nor smaller.
		if(this_mmd_process < md_batch_n_processes*n_md_batches)
			md_batch_pcolor = int(this_mmd_process/md_batch_n_processes);
		// Initially we used MPI_UNDEFINED, but why waste processes... The processes over
		// 'n_lammps_processes_per_batch*n_lammps_batch' are assigned to the last batch...
		// finally it is better to waste them than failing the simulation with an odd number
		// of processes for the last batch
		/*else
			mmd_pcolor = int((md_batch_n_processes*n_md_batches-1)/md_batch_n_processes);
		*/

		// Definition of the communicators
		MPI_Comm_split(mmd_communicator, md_batch_pcolor, this_mmd_process, &md_batch_communicator);
		MPI_Comm_rank(md_batch_communicator,&this_md_batch_process);

	}



	template <int dim>
	void MMDProblem<dim>::prepare_md_simulations()
	{
		// Check list of files corresponding to current "time_id"
		ncupd = 0;
		char filenamelist[1024];
		sprintf(filenamelist, "%s/last.qpupdates", macrostatelocout.c_str());
		std::ifstream ifile;
		std::string iline;

		// Count number of cells to update
		ifile.open (filenamelist);
		if (ifile.is_open())
		{
			while (getline(ifile, iline)) ncupd++;
			ifile.close();
		}
		else mcout << "Unable to open" << filenamelist << " to read it" << std::endl;

		if (ncupd>0){
			// Create list of quadid
			cell_id.resize(ncupd,"");
			ifile.open (filenamelist);
			int nline = 0;
			while (nline<ncupd && std::getline(ifile, cell_id[nline])) nline++;
			ifile.close();

			// Load material type of cells to be updated
			cell_mat.resize(ncupd,"");
			sprintf(filenamelist, "%s/last.matqpupdates", macrostatelocout.c_str());
			ifile.open (filenamelist);
			nline = 0;
			while (nline<ncupd && std::getline(ifile, cell_mat[nline])) nline++;
			ifile.close();

			// Number of MD simulations at this iteration...
			int nmdruns = ncupd*nrepl;

			// Location of each MD simulation temporary log files
			qpreplogloc.resize(nmdruns,"");

			// Setting up batch of processes
			set_md_procs(nmdruns);

			// Preparing strain input file for each replica
			for (int c=0; c<ncupd; ++c)
			{
				int imd = 0;
				for(unsigned int i=0; i<mdtype.size(); i++)
					if(cell_mat[c]==mdtype[i])
						imd=i;

				for(unsigned int repl=0;repl<nrepl;repl++)
				{
					// Offset replica number because in filenames, replicas start at 1
					int numrepl = repl+1;

					// The variable 'imdrun' assigned to a run is a multiple of the batch number the run will be run on
					int imdrun=c*nrepl + (repl);

					// Allocation of a MD run to a batch of processes
					if (md_batch_pcolor == (imdrun%n_md_batches)){
						// Setting up location for temporary log outputs of md simulation
						qpreplogloc[imdrun] = nanologloctmp + "/" + time_id  + "." + cell_id[c] + "." + cell_mat[c] + "_" + std::to_string(numrepl);

						if(this_md_batch_process == 0){

							SymmetricTensor<2,dim> loc_rep_strain, cg_loc_rep_strain;

							char filename[1024];

							// Argument of the MD simulation: strain to apply
							sprintf(filename, "%s/last.%s.upstrain", macrostatelocout.c_str(), cell_id[c].c_str());
							read_tensor<dim>(filename, cg_loc_rep_strain);

							// Rotate strain tensor from common ground to replica orientation
							loc_rep_strain = rotate_tensor(cg_loc_rep_strain, transpose(replica_data[imd*nrepl+repl].rotam));

							// Write tensor to replica specific file
							sprintf(filename, "%s/last.%s.%d.upstrain", macrostatelocout.c_str(), cell_id[c].c_str(), numrepl);
							write_tensor<dim>(filename, loc_rep_strain);

							// Preparing directory to write MD simulation log files
							mkdir(qpreplogloc[imdrun].c_str(), ACCESSPERMS);
						}
					}
				}
			}
		}
	}



	template <int dim>
	void MMDProblem<dim>::execute_inside_md_simulations()
	{
		// Computing cell state update running one simulation per MD replica (basic job scheduling and executing)
		mcout << "        " << "...dispatching the MD runs on batch of processes..." << std::endl;
		mcout << "        " << "...cells and replicas completed: " << std::flush;
		for (int c=0; c<ncupd; ++c)
		{
			for(unsigned int repl=0;repl<nrepl;repl++)
			{
				// Offset replica number because in filenames, replicas start at 1
				int numrepl = repl+1;

				// The variable 'imdrun' assigned to a run is a multiple of the batch number the run will be run on
				int imdrun=c*nrepl + (repl);

				// Allocation of a MD run to a batch of processes
				if (md_batch_pcolor == (imdrun%n_md_batches)){

					std::string command = "mpirun ./single_md"
											+" "+cell_id[c]+" "+time_id+" "+cell_mat[c]
											+" "+nanostatelocout+" "+nanostatelocres+" "+nanologlochom
											+" "+qpreplogloc[imdrun]+" "+md_scripts_directory+" "+macrostatelocout
											+" "+std::to_string(repl)+" "+md_timestep_length+" "+md_temperature
											+" "+md_nsteps_sample+" "+md_strain_rate+" "+output_homog
											+" "+checkpoint_save;

					int ret = system(command.c_str());
					if (ret!=0){
						std::cerr << "Failed executing the md simulation: " << command << std::endl;
						exit(1);
					}
					//run_single_md(time_id, cell_id[c].c_str(), cell_mat[c].c_str(), numrepl, qpreplogloc[imdrun]);
				}
			}
		}
		mcout << std::endl;
	}



	template <int dim>
	void MMDProblem<dim>::store_md_simulations()
	{
		// Averaging stiffness and stress per cell over replicas
		for (int c=0; c<ncupd; ++c)
		{
			int imd = 0;
			for(unsigned int i=0; i<mdtype.size(); i++)
				if(cell_mat[c]==mdtype[i])
					imd=i;

			if (md_batch_pcolor == (c%n_md_batches))
			{
				// Write the new stress and stiffness tensors into two files, respectively
				// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
				if(this_md_batch_process == 0)
				{
					//SymmetricTensor<4,dim> cg_loc_stiffness;
					SymmetricTensor<2,dim> cg_loc_stress;
					char filename[1024];

					for(unsigned int repl=0;repl<nrepl;repl++)
					{
						// Offset replica number because in filenames, replicas start at 1
						int numrepl = repl+1;

						// Rotate stress and stiffness tensor from replica orientation to common ground

						//SymmetricTensor<4,dim> cg_loc_stiffness, loc_rep_stiffness;
						SymmetricTensor<2,dim> cg_loc_rep_stress, loc_rep_stress;
						sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout.c_str(), cell_id[c].c_str(), numrepl);

						if(read_tensor<dim>(filename, loc_rep_stress)){
							/* // Rotation of the stiffness tensor to common ground direction before averaging
							sprintf(filename, "%s/last.%s.%d.stiff", macrostatelocout.c_str(), cell_id[c], repl);
							read_tensor<dim>(filename, loc_rep_stiffness);

							cg_loc_stiffness = rotate_tensor(loc_stiffness, replica_data[imd*nrepl+repl].rotam);

							cg_loc_stiffness += cg_loc_rep_stiffness;*/

							// Rotation of the stress tensor to common ground direction before averaging
							cg_loc_rep_stress = rotate_tensor(loc_rep_stress, replica_data[imd*nrepl+repl].rotam);

							cg_loc_stress += cg_loc_rep_stress;

							// Removing file now it has been used
							remove(filename);

							// Removing replica strain passing file used to average cell stress
							sprintf(filename, "%s/last.%s.%d.upstrain", macrostatelocout.c_str(), cell_id[c].c_str(), numrepl);
							remove(filename);

							std::string qpreplogloc = nanologloctmp + "/" + time_id  + "." + cell_id[c] + "." + cell_mat[c] + "_" + std::to_string(numrepl);

							// Clean "nanoscale_logs" of the finished timestep
							char command[1024];
							sprintf(command, "rm -rf %s", qpreplogloc.c_str());
							int ret = system(command);
							if (ret!=0){
								std::cerr << "Failed removing the log files of the MD simulations of the current step!" << std::endl;
								exit(1);
							}
						}
					}

					//cg_loc_stiffness /= nrepl;
					cg_loc_stress /= nrepl;

					/*sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), cell_id[c].c_str());
					write_tensor<dim>(filename, cg_loc_stiffness);*/

					sprintf(filename, "%s/last.%s.stress", macrostatelocout.c_str(), cell_id[c].c_str());
					write_tensor<dim>(filename, cg_loc_stress);
				}
			}
		}

	}


	template <int dim>
	void MMDProblem<dim>::init_mmd (int sstp, double mdtlength, double mdtemp, int nss, double strr,
			   std::string nslocin, std::string nslocout, std::string nslocres, std::string nlogloc,
			   std::string nlogloctmp,std::string nloglochom, std::string mslocout, std::string mdsdir,
			   int fchpt, int fohom, unsigned int bnmin, unsigned int mppn,
			   std::vector<std::string> mdt, Tensor<1,dim> cgd, unsigned int nr, bool ups){

		start_timestep = sstp;

		md_timestep_length = mdtlength;
		md_temperature = mdtemp;
		md_nsteps_sample = nss;
		md_strain_rate = strr;

		nanostatelocin = nslocin;
		nanostatelocout = nslocout;
		nanostatelocres = nslocres;
		nanologloc = nlogloc;
		nanologloctmp = nlogloctmp;
		nanologlochom = nloglochom;

		macrostatelocout = mslocout;
		md_scripts_directory = mdsdir;

		freq_checkpoint = fchpt;
		freq_output_homog = fohom;

		batch_nnodes_min = bnmin;
		machine_ppn = mppn;

		mdtype = mdt;
		cg_dir = cgd;
		nrepl = nr;

		use_pjm_scheduler = ups;

		restart ();
		initialize_replicas ();
	}

	template <int dim>
	void MMDProblem<dim>::update_mmd (int tstp, double ptime, int nstp){
		present_time = ptime;
		timestep = tstp;
		newtonstep = nstp;

		cell_id.clear();
		cell_mat.clear();
		time_id = std::to_string(timestep)+"-"+std::to_string(newtonstep);
		qpreplogloc.clear();

		// Should the homogenization trajectory file be saved?
		if (timestep%freq_output_homog==0) output_homog = true;
		else output_homog = false;

		if (timestep%freq_output_homog==0) checkpoint_save = true;
		else checkpoint_save = false;

		prepare_md_simulations();
		MPI_Barrier(mmd_communicator);
		if (ncupd>0){
			if(use_pjm_scheduler){
				execute_pjm_md_simulations();
			}
			else{
				execute_inside_md_simulations();
			}

			MPI_Barrier(mmd_communicator);
			store_md_simulations();
		}
	}





	template <int dim>
	class HMMProblem
	{
	public:
		HMMProblem ();
		~HMMProblem ();
		void run (std::string inputfile);

	private:
		void read_inputs(std::string inputfile);

		void set_global_communicators ();
		void setup_replica_data ();
		void set_repositories ();


		void solve_timestep ();
		void do_timestep ();

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		MPI_Comm 							fe_communicator;
		int									root_fe_process;
		int 								n_fe_processes;
		int 								this_fe_process;
		int 								fe_pcolor;

		MPI_Comm 							mmd_communicator;
		int 								n_mmd_processes;
		int									root_mmd_process;
		int 								this_mmd_process;
		int 								mmd_pcolor;

		unsigned int						machine_ppn;
		int									fenodes;
		unsigned int						batch_nnodes_min;

		ConditionalOStream 					hcout;

		int									start_timestep;
		int									end_timestep;
		double              				present_time;
		double              				fe_timestep_length;
		double              				end_time;
		int        							timestep;
		int        							newtonstep;

		int									fe_degree;
		int									quadrature_formula;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;
		Tensor<1,dim> 						cg_dir;

		bool								activate_md_update;
		bool								use_pjm_scheduler;

		double								md_timestep_length;
		double								md_temperature;
		int									md_nsteps_sample;
		double								md_strain_rate;

		int									freq_checkpoint;
		int									freq_output_visu;
		int									freq_output_lhist;
		int									freq_output_homog;

		std::string                         macrostatelocin;
		std::string                         macrostatelocout;
		std::string							macrostatelocres;
		std::string							macrologloc;

		std::string                         nanostatelocin;
		std::string							nanostatelocout;
		std::string							nanostatelocres;
		std::string							nanologloc;
		std::string							nanologloctmp;
		std::string							nanologlochom;

		std::string							md_scripts_directory;
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
	void HMMProblem<dim>::read_inputs (std::string inputfile)
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

	    // Continuum timestepping
	    fe_timestep_length = std::stof(bptree_read(pt, "continuum time", "timestep length"));
	    start_timestep = std::stof(bptree_read(pt, "continuum time", "start timestep"));
	    end_timestep = std::stof(bptree_read(pt, "continuum time", "end timestep"));

	    // Continuum meshing
	    fe_degree = std::stoi(bptree_read(pt, "continuum mesh", "fe degree"));
	    quadrature_formula = std::stoi(bptree_read(pt, "continuum mesh", "quadrature formula"));

	    // Scale-bridging parameters
	    activate_md_update = std::stoi(bptree_read(pt, "scale-bridging", "activate md update"));
	    use_pjm_scheduler = std::stoi(bptree_read(pt, "scale-bridging", "use pjm scheduler"));

	    // Continuum input, output, restart and log location
		macrostatelocin = bptree_read(pt, "directory structure", "macroscale input");
		macrostatelocout = bptree_read(pt, "directory structure", "macroscale output");
		macrostatelocres = bptree_read(pt, "directory structure", "macroscale restart");
		macrologloc= bptree_read(pt, "directory structure", "macroscale log");

		// Atomic input, output, restart and log location
		nanostatelocin = bptree_read(pt, "directory structure", "nanoscale input");
		nanostatelocout = bptree_read(pt, "directory structure", "nanoscale output");
		nanostatelocres = bptree_read(pt, "directory structure", "nanoscale restart");
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
			tmp_dir.push_back(v.second.data());
		}
		if(tmp_dir.size()==cg_dir.size()) cg_dir = tmp_dir;

		// Molecular dynamics simulation parameters
		md_timestep_length = std::stod(bptree_read(pt, "molecular dynamics parameters", "timestep length"));
		md_temperature = std::stod(bptree_read(pt, "molecular dynamics parameters", "temperature"));
		md_nsteps_sample = std::stoi(bptree_read(pt, "molecular dynamics parameters", "number of sampling steps"));
		md_strain_rate = std::stod(bptree_read(pt, "molecular dynamics parameters", "strain rate"));
		md_scripts_directory = bptree_read(pt, "molecular dynamics parameters", "scripts directory");

		// Computational resources
		machine_ppn = std::stoi(bptree_read(pt, "computational resources", "machine cores per node"));
		fenodes = std::stoi(bptree_read(pt, "computational resources", "number of nodes for FEM simulation"));
		batch_nnodes_min = std::stoi(bptree_read(pt, "computational resources", "minimum nodes per MD simulation"));

		// Output and checkpointing frequencies
		freq_checkpoint = std::stoi(bptree_read(pt, "output data", "checkpoint frequency"));
		freq_output_lhist = std::stoi(bptree_read(pt, "output data", "visualisation output frequency"));
		freq_output_visu = std::stoi(bptree_read(pt, "output data", "analytics output frequency"));
		freq_output_homog = std::stoi(bptree_read(pt, "output data", "homogenization output frequency"));

		// Print a recap of all the parameters...
		hcout << "Parameters listing:" << std::endl;
		hcout << " - Activate MD updates (1 is true, 0 is false): "<< activate_md_update << std::endl;
		hcout << " - Use Pilor Job Manager to schedule MD jobs: "<< use_pjm_scheduler << std::endl;
		hcout << " - FE timestep duration: "<< fe_timestep_length << std::endl;
		hcout << " - Start timestep: "<< start_timestep << std::endl;
		hcout << " - End timestep: "<< end_timestep << std::endl;
		hcout << " - FE shape funciton degree: "<< fe_degree << std::endl;
		hcout << " - FE quadrature formula: "<< quadrature_formula << std::endl;
		hcout << " - Number of replicas: "<< nrepl << std::endl;
		hcout << " - List of material names: "<< std::flush;
		for(unsigned int imd=0; imd<mdtype.size(); imd++) hcout << " " << mdtype[imd] << std::flush; hcout << std::endl;;
		hcout << " - MD timestep duration: "<< md_timestep_length << std::endl;
		hcout << " - MD thermostat temperature: "<< md_temperature << std::endl;
		hcout << " - MD deformation rate: "<< md_strain_rate << std::endl;
		hcout << " - MD number of sampling steps: "<< md_nsteps_sample << std::endl;
		hcout << " - MD scripts directory (contains in.set, in.strain, ELASTIC/, ffield parameters): "<< md_scripts_directory << std::endl;
		hcout << " - Number of cores per node on the machine: "<< machine_ppn << std::endl;
		hcout << " - Number of nodes for FEM simulation: "<< fenodes << std::endl;
		hcout << " - Minimum number of nodes per MD simulation: "<< bnnodes_min << std::endl;
		hcout << " - Frequency of checkpointing: "<< freq_checkpoint << std::endl;
		hcout << " - Frequency of writing FE data files: "<< freq_output_lhist << std::endl;
		hcout << " - Frequency of writing FE visualisation files: "<< freq_output_visu << std::endl;
		hcout << " - Frequency of writing MD homogenization trajectory files: "<< freq_output_homog << std::endl;
		hcout << " - FE input directory: "<< macrostatelocin << std::endl;
		hcout << " - FE output directory: "<< macrostatelocout << std::endl;
		hcout << " - FE restart directory: "<< macrostatelocres << std::endl;
		hcout << " - FE log directory: "<< macrologloc << std::endl;
		hcout << " - MD input directory: "<< nanostatelocin << std::endl;
		hcout << " - MD output directory: "<< nanostatelocout << std::endl;
		hcout << " - MD restart directory: "<< nanostatelocres << std::endl;
		hcout << " - MD log directory: "<< nanologloc << std::endl;
	}




	template <int dim>
	void HMMProblem<dim>::set_global_communicators ()
	{
		//Setting up DEALII communicator and related variables
		root_fe_process = 0;
		n_fe_processes = fenodes*machine_ppn;
		// Color set above 0 for processors that are going to be used
		fe_pcolor = MPI_UNDEFINED;
		if (this_world_process >= root_fe_process &&
				this_world_process < root_fe_process + n_fe_processes) fe_pcolor = 0;
		else fe_pcolor = 1;

		MPI_Comm_split(world_communicator, fe_pcolor, this_world_process, &fe_communicator);
		MPI_Comm_rank(fe_communicator, &this_fe_process);


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
	void HMMProblem<dim>::set_repositories ()
	{
		if(!file_exists(macrostatelocin) || !file_exists(nanostatelocin)){
			std::cerr << "Missing macroscale or nanoscale input directories." << std::endl;
			exit(1);
		}

		mkdir(macrostatelocout.c_str(), ACCESSPERMS);
		mkdir(macrostatelocres.c_str(), ACCESSPERMS);
		mkdir(macrologloc.c_str(), ACCESSPERMS);

		mkdir(nanostatelocout.c_str(), ACCESSPERMS);
		mkdir(nanostatelocres.c_str(), ACCESSPERMS);
		mkdir(nanologloc.c_str(), ACCESSPERMS);
		nanologloctmp = nanologloc+"/tmp"; mkdir(nanologloctmp.c_str(), ACCESSPERMS);
		nanologlochom = nanologloc+"/homog"; mkdir(nanologlochom.c_str(), ACCESSPERMS);

		char fnset[1024]; sprintf(fnset, "%s/in.set.lammps", md_scripts_directory.c_str());
		char fnstrain[1024]; sprintf(fnstrain, "%s/in.strain.lammps", md_scripts_directory.c_str());
		char fnelastic[1024]; sprintf(fnelastic, "%s/ELASTIC", md_scripts_directory.c_str());

		if(!file_exists(fnset) || !file_exists(fnstrain) || !file_exists(fnelastic)){
			std::cerr << "Missing some MD input scripts for executing LAMMPS simulation (in 'box' directory)." << std::endl;
			exit(1);
		}
	}




	template <int dim>
	void HMMProblem<dim>::do_timestep (FEProblem<dim> &fe_problem)
	{
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
		if(fe_pcolor==0) fe_problem.beginstep(timestep, present_time);
		MPI_Barrier(world_communicator);

		// Solving iteratively the current timestep
		bool continue_newton = false;

		do
		{
			++newtonstep;

			if(fe_pcolor==0) fe_problem.solve(newtonstep);

			MPI_Barrier(world_communicator);

			if(mmd_pcolor==0) mmd_problem.update_mmd(timestep, present_time, newtonstep);
			MPI_Barrier(world_communicator);

			if(fe_pcolor==0) continue_newton = fe_problem.check();

			// Share the value of previous_res with processors outside of dealii allocation
			MPI_Bcast(&continue_newton, 1, MPI_BOOL, root_fe_process, world_communicator);

		} while (continue_newton);

		if(fe_pcolor==0) fe_problem.endstep();

		MPI_Barrier(world_communicator);
	}



	template <int dim>
	void HMMProblem<dim>::run (std::string inputfile)
	{
		// Reading JSON input file (common ones only)
		read_inputs(inputfile);

		hcout << "Building the HMM problem:       " << std::endl;

		// Set the dealii communicator using a limited amount of available processors
		// because dealii fails if processors do not have assigned cells. Plus, dealii
		// might not scale indefinitely
		set_global_communicators();

		// Instantiation of the MMD Proble
		if(mmd_pcolor==0) MMDProblem<dim> mmd_problem (mmd_communicator, mmd_pcolor);

		// Instantiation of the FE problem
		if(fe_pcolor==0) FEProblem<dim> fe_problem (fe_communicator, fe_pcolor, fe_degree, quadrature_formula);
		MPI_Barrier(world_communicator);

		// Setting repositories for input and creating repositories for outputs
		set_repositories();

		// Setup replicas information vector
		setup_replica_data();

		MPI_Barrier(world_communicator);

		// Initialization of time variables
		timestep = start_timestep - 1;
		present_time = timestep*fe_timestep_length;
		end_time = end_timestep*fe_timestep_length; //4000.0 > 66% final strain

		hcout << " Initialization of the Multiple Molecular Dynamics problem...       " << std::endl;
		if(mmd_pcolor==0) mmd_problem.init_mmd(start_timestep, md_timestep_length, md_temperature,
											   md_nsteps_sample, md_strain_rate, nanostatelocin,
											   nanostatelocout, nanostatelocres, nanologloc,
											   nanologloctmp, nanologlochom, macrostatelocout,
											   md_scripts_directory, freq_checkpoint, freq_output_homog,
											   batch_nnodes_min, machine_ppn, mdtype, cg_dir, nrepl,
											   use_pjm_scheduler);

		// Initialization of MMD must be done before initialization of FE, because FE needs initial
		// materials properties obtained from MMD initialization

		hcout << " Initiation of the Finite Element problem...       " << std::endl;
		MPI_Barrier(world_communicator);
		if(fe_pcolor==0) fe_problem.init(start_timestep, fe_timestep_length,
										 macrostatelocin, macrostatelocout,
										 macrostatelocres, macrologloc,
										 freq_checkpoint, freq_output_visu, freq_output_lhist,
										 activate_md_update, mdtype, cg_dir);

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

		if(argc!=2){
			std::cerr << "Wrong number of arguments, expected: './dealammps inputs_hmm.json', but argc is " << argc << std::endl;
			exit(1);
		}

		std::string inputfile = argv[1];
		if(!file_exists(inputfile)){
			std::cerr << "Missing HMM input file." << std::endl;
			exit(1);
		}

		HMMProblem<3> hmm_problem;
		hmm_problem.run(inputfile);
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
