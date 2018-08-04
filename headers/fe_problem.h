#ifndef FE_PROBLEM_H
#define FE_PROBLEM_H


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
#include "read_write.h"
#include "tensor_calc.h"

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
		FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for);
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
		(const Vector<double>& displacement_update);
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
	FEProblem<dim>::FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for)
	:
		FE_communicator (dcomm),
		n_FE_processes (Utilities::MPI::n_mpi_processes(FE_communicator)),
		this_FE_process (Utilities::MPI::this_mpi_process(FE_communicator)),
		FE_pcolor (pcolor),
		dcout (std::cout,(this_FE_process == 0)),
		triangulation(FE_communicator),
		dof_handler (triangulation),
		fe (FE_Q<dim>(fe_deg), dim),
		quadrature_formula (quad_for)
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
			dcout << "       material: " << mdtype[imd].c_str() << std::endl;

			// Reading initial material stiffness tensor
			sprintf(filename, "%s/init.%s.stiff", macrostatelocout.c_str(), mdtype[imd].c_str());
			read_tensor<dim>(filename, stiffness_tensors[imd]);

			if(this_FE_process==0){
				std::cout << "          * stiffness: " << std::endl;
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

			dcout << "          * density: " << densities[imd] << std::endl;

			sprintf(filename, "%s/last.%s.density", macrostatelocout.c_str(), mdtype[imd].c_str());
				write_tensor<dim>(filename, densities[imd]);


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
				// (iii) cells based on their id
				if (activate_md_update)
				//if (activate_md_update && cell->barycenter()(1) <  3.0*tt && cell->barycenter()(0) <  1.10*(ww - aa) && cell->barycenter()(0) > 0.0*(ww - aa))
				/*if (activate_md_update && (cell->active_cell_index() == 2922 || cell->active_cell_index() == 2923
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
			std::string filename = macrologloc + "/" + "solution-" + Utilities::int_to_string(timestep,4)
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
				visit_master_filename = (macrologloc + "/" + "solution-" +
						Utilities::int_to_string(timestep,4) +
						".visit");
				std::ofstream visit_master (visit_master_filename.c_str());
				//data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
				DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

				const std::string
				pvtu_master_filename = (macrologloc + "/" + "solution-" +
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
				std::ofstream pvd_output (macrologloc + "/" + "solution.pvd");
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
		if(timestep%freq_checkpoint==0){
			char timeid[1024];
			sprintf(timeid, "%s", "lcts");
			checkpoint (timeid);
			sprintf(timeid, "%d", timestep);
			checkpoint (timeid);
		}

		dcout << std::endl;
	}
}

#endif
