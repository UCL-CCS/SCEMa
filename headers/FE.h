#ifndef FE_H
#define FE_H


#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>
#include <numeric>
#include <random>

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "read_write.h"
#include "math_calc.h"
#include "scale_bridging_data.h"

// Reduction model based on spline comparison
#include "strain2spline.h"

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
#include <deal.II/grid/grid_out.h>
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
		MatHistPredict::Strain6D hist_strain;
		bool to_be_updated;

		// Characteristics
		unsigned int qpid;
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
	class CellData {
		public:
			CellData()
			{
			}
			
			std::vector<int> composition;

			void generate_nanostructure_uniform(
					parallel::shared::Triangulation<dim>& triangulation,
					std::vector<double> proportions)
			{
				// check proportions of materials add up to 1
				const double epsilon = 0.0001;
				double sum = std::accumulate(proportions.begin(), 
							     proportions.end(), 0.0);
				if (fabs(1.0 - sum) > epsilon)
				{
					std::cout << "Material proprtions must sum to 1"<< std::endl;
					exit(1);
				}

				// random number generator 
				std::mt19937 generator (time(0));
				std::uniform_real_distribution<double> dist(0.0, 1.0);

				// for each cell asign a material type based on the proportion
				for (unsigned int cell=0; cell < triangulation.n_active_cells(); cell++)
				{
						double r = dist(generator);
						double k = 0;
						for (unsigned int i=0; i < proportions.size(); i++)
						{
								k += proportions[i];
								if (k > r)
								{
										composition.push_back(i);
										break;
								}	
						}
				}
			}

			int get_composition(int cell_index)
			{
					return composition[cell_index];
			}

			int number_of_boxes()
			{ 	
					return composition.size();
			}
		private:
			//std::vector<Vector> coords;
			//std::vector<Vector> normal;
	};


	template <int dim>
			class FEProblem
			{
					public:
							FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for, const int n_world_processes);
							~FEProblem ();

							void init (int sstp, double tlength, std::string mslocin, std::string mslocout,
											std::string mslocres, std::string mlogloc, int fchpt, int fovis, int folhis,
											bool actmdup, std::vector<std::string> mdt, Tensor<1,dim> cgd, 
											std::string twodmfile, double extrudel, int extrudep, 
											boost::property_tree::ptree inconfig, bool hookeslaw);
							void beginstep (int tstp, double ptime);
							void solve (int nstp, ScaleBridgingData &scale_bridging_data);
							bool check (ScaleBridgingData scale_bridging_data);
							void endstep ();

					private:
							void make_grid ();
						  void visualise_mesh(parallel::shared::Triangulation<dim> &triangulation);
							void setup_system ();
							CellData<dim> get_microstructure ();
							std::vector<Vector<double> > generate_microstructure_uniform();
							void assign_microstructure (typename DoFHandler<dim>::active_cell_iterator cell, 
											CellData<dim> celldata,
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
							void check_strain_quadrature_point_history();
							void spline_building();
							void spline_comparison();
							void history_analysis();
							void write_md_updates_list(ScaleBridgingData &scale_bridging_data);

							void gather_qp_update_list(ScaleBridgingData &scale_bridging_data);
							template <typename T>
							std::vector<T> gather_vector(std::vector<T> local_vector);
							void update_stress_quadrature_point_history
									(const Vector<double>& displacement_update, ScaleBridgingData scale_bridging_data);
							void clean_transfer();

							Vector<double>  compute_internal_forces () const;
							std::vector< std::vector< Vector<double> > >
									compute_history_projection_from_qp_to_nodes (FE_DGQ<dim> &history_fe, DoFHandler<dim> &history_dof_handler, std::string stensor) const;
							void output_lhistory ();
							void output_visualisation_solution ();
							void output_visualisation_history ();
							void output_results ();
							void checkpoint () const;

							Vector<double> 		     			newton_update_displacement;
							Vector<double> 		     			incremental_displacement;
							Vector<double> 		     			displacement;
							Vector<double> 		     			old_displacement;

							Vector<double> 		     			newton_update_velocity;
							Vector<double> 		     			incremental_velocity;
							Vector<double> 		     			velocity;
							//Vector<double> 		     		old_velocity;

							MPI_Comm 							FE_communicator;
							unsigned int 							n_FE_processes;
							int 								this_FE_process;
							int								root_FE_process;
							int 								FE_pcolor;
							unsigned int							n_world_processes;

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

							FE_DGQ<dim>     					history_fe;
							DoFHandler<dim> 					history_dof_handler;

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

							double 								ll;
							double 								lls;
							double 								ww;
							double 								bb;
							double								cc;
							double								diam_weight;

							std::vector<std::string> 			mdtype;
							Tensor<1,dim> 						cg_dir;

							int 								num_spline_points;
							int 								min_num_steps_before_spline;
							double								acceptable_diff_threshold;
							std::string							splinescriptsloc;

							std::string                         macrostatelocin;
							std::string                         macrostatelocout;
							std::string                         macrostatelocres;
							std::string                         macrologloc;

							int									freq_checkpoint;
							int									freq_output_visu;
							int									freq_output_lhist;

							bool 								activate_md_update;

							std::string		twod_mesh_file;
							double                  extrude_length;
							int                     extrude_points;

							boost::property_tree::ptree     input_config;
							bool														approx_md_with_hookes_law;
							
							CellData<dim> celldata;

							ProblemType<dim>* problem_type = NULL;
			};
}
#endif    
