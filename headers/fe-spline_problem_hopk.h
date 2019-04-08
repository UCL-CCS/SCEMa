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
#include "tensor_calc.h"
#include "scale_bridging_data.h"
#include "problem_type.h"

// Reduction model based on spline comparison
#include "../spline/strain2spline.h"

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



	template <int dim>
			FEProblem<dim>::FEProblem (MPI_Comm dcomm, int pcolor, int fe_deg, int quad_for, 
							const int n_total_processes)
			:
					n_world_processes (n_total_processes),
					FE_communicator (dcomm),
					n_FE_processes (Utilities::MPI::n_mpi_processes(FE_communicator)),
					this_FE_process (Utilities::MPI::this_mpi_process(FE_communicator)),
					FE_pcolor (pcolor),
					dcout (std::cout,(this_FE_process == 0)),
					triangulation(FE_communicator),
					dof_handler (triangulation),
					fe (FE_Q<dim>(fe_deg), dim),
					quadrature_formula (quad_for),
					history_fe (1),
					history_dof_handler (triangulation)
		{}



	template <int dim>
			FEProblem<dim>::~FEProblem ()
			{
					dof_handler.clear ();
			}

	template <int dim>
			void FEProblem<dim>::make_grid ()
			{
					std::string mesh_input_style;
					mesh_input_style    = input_config.get<std::string>("continuum mesh.input.style");
					
					if (mesh_input_style == "cuboid")
					{
						problem_type = (ProblemType<dim>*) new DropWeight<dim>(input_config);
						problem_type->make_grid(triangulation);
					}
					else if (mesh_input_style == "file")
					{
						std::string         twod_mesh_file;
		                double              extrude_length;
        		        int                 extrude_points;

						twod_mesh_file    = input_config.get<std::string>("continuum mesh.input.2D mesh file");
        				extrude_length    = input_config.get<double>( "continuum mesh.input.extrude length");
        				extrude_points    = input_config.get<int>("continuum mesh.input.extrude points");

						char filename[1024];
						sprintf(filename, "%s/%s", macrostatelocin.c_str(), twod_mesh_file.c_str());
						std::ifstream iss(filename);
						if (iss.is_open()){

							dcout << "    Reading in 2D mesh" << std::endl;
							Triangulation<2> triangulation2D;
							GridIn<2> gridin;
							gridin.attach_triangulation(triangulation2D);
							sprintf(filename, "%s/%s", macrostatelocin.c_str(), twod_mesh_file.c_str());
							std::ifstream f(filename);
							gridin.read_msh(f);

							dcout << "    extruding by " << extrude_length;
							dcout << " with "<< extrude_points << " points" << std::endl; 
							GridGenerator::extrude_triangulation (triangulation2D, extrude_points, extrude_length, triangulation);
						}
					}

					// Check that the FEM is not passed less ranks than cells
					if ( triangulation.n_active_cells() < n_FE_processes &&
									triangulation.n_active_cells() < n_world_processes ){
							dcout << "Exception: Cells < ranks in FE communicator... " << std::endl;
							exit(1);
					}

					visualise_mesh(triangulation);

					// Saving triangulation, not usefull now and costly...
					//sprintf(filename, "%s/mesh.tria", macrostatelocout.c_str());
					//std::ofstream oss(filename);
					//boost::archive::text_oarchive oa(oss, boost::archive::no_header);
					//triangulation.save(oa, 0);
							
					dcout << "    Number of active cells:       "
							<< triangulation.n_active_cells()
							<< " (by partition:";
					for (unsigned int p=0; p<n_FE_processes; ++p)
							dcout << (p==0 ? ' ' : '+')
									<< (GridTools::
													count_cells_with_subdomain_association (triangulation,p));
					dcout << ")" << std::endl;
			}

	template<int dim>
	void FEProblem<dim>::visualise_mesh(parallel::shared::Triangulation<dim> &triangulation)
	{
	if (this_FE_process==0){
		char filename[1024];
		sprintf(filename, "%s/3D_mesh.eps", macrostatelocout.c_str());
		std::ofstream out (filename);
		GridOut grid_out;
		grid_out.write_eps (triangulation, out);
		dcout << "    mesh .eps written to " << filename << std::endl;	
		}
	}




	template <int dim>
			void FEProblem<dim>::setup_system ()
			{
					dof_handler.distribute_dofs (fe);
					locally_owned_dofs = dof_handler.locally_owned_dofs();
					DoFTools::extract_locally_relevant_dofs (dof_handler,locally_relevant_dofs);

					history_dof_handler.distribute_dofs (history_fe);

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
					for (unsigned int p=0; p<n_FE_processes; ++p)
							dcout << (p==0 ? ' ' : '+')
									<< (DoFTools::
													count_dofs_with_subdomain_association (dof_handler,p));
					dcout << ")" << std::endl;
			}



	template <int dim>
			CellData<dim> FEProblem<dim>::get_microstructure ()
			{
					std::string 	distribution_type;

					distribution_type = input_config.get<std::string>("molecular dynamics material.distribution.style");
					if (distribution_type == "uniform"){
							dcout << " generating uniform distribution of materials... " << std::endl;

							std::vector<double> proportions;
							BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
											input_config.get_child("molecular dynamics material.distribution.proportions.")) 
							{
									proportions.push_back(std::stod(v.second.data()));
							}

							// check length of materials list and proportions list are the same
							if (mdtype.size() != proportions.size())
							{
									dcout<< "Materials list and proportions list must be the same length" <<std::endl;
									exit(1);
							}
							// Generate nanostructure on rank 0, then broadcast it to other ranks
							if (this_FE_process == 0){
								celldata.generate_nanostructure_uniform(triangulation, proportions);
							}
							else {
								celldata.composition.resize(triangulation.n_active_cells());
							}
							MPI_Bcast(&(celldata.composition[0]), triangulation.n_active_cells(), MPI_INT, 0, FE_communicator);
							/*dcout<<"CHECK"<<std::endl;
							if (this_FE_process == 0){
								for (int i = 0; i< 10; i++){
									std::cout<< celldata.composition[i] << " ";
								}
								std::cout <<std::endl;
							}
							if (this_FE_process == 1){		
								for (int i = 0; i< 10; i++){
									std::cout<< celldata.composition[i] << " ";
								}
								std::cout <<std::endl;
							}*/	
					}		
					/*unsigned int npoints = 0;
					  unsigned int nfchar = 0;
					  std::vector<Vector<double> > structure_data (npoints, Vector<double>(nfchar)); 
					  else if (distribution_type == "file"){
					// this is maxime's method of populating structure_data from a file

					// Load flakes data (center position, angles, density)

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
					}*/

					return celldata;
			}


	template <int dim>
			void FEProblem<dim>::assign_microstructure (typename DoFHandler<dim>::active_cell_iterator cell, CellData<dim> celldata,
							std::string &mat, Tensor<2,dim> &rotam)
			{

					// Filling identity matrix
					Tensor<2,dim> idmat;
					idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;

					// Default orientation of cell
					rotam = idmat;

					unsigned int n = cell->active_cell_index();
					mat = mdtype[ celldata.get_composition(n) ];	
					//std::cout << n << " " << mat <<" "<<celldata.get_composition(n)<< std::endl;

					/*
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
					 */

					//std::cout << " box number: " << n << " is in cell " << cell->active_cell_index()
					//  		  << " of material " << mat << std::endl;

					// Assembling the rotation matrix from the global orientation of the cell given by the
					// microstructure to the common ground direction
					//rotam = compute_rotation_tensor(nglo, cg_dir);

					// Stop the for loop since a cell can only be in one flake at a time...
					//break;
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

							// Create file with mdtype of qptid to update at timeid
							std::ofstream omatfile;
							char mat_local_filename[1024];
							sprintf(mat_local_filename, "%s/cell_id_mat.%d.list", macrostatelocout.c_str(), this_FE_process);
							omatfile.open (mat_local_filename);

							// Load the microstructure
							dcout << "    Loading microstructure..." << std::endl;
							celldata = get_microstructure();

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
													local_quadrature_points_history[q].qpid = cell->active_cell_index()*quadrature_formula.size() + q;

													// Tell strain history object what cell ID it belongs to
													local_quadrature_points_history[q].hist_strain.set_ID(local_quadrature_points_history[q].qpid);

													// Assign microstructure to the current cell (so far, mdtype
													// and rotation from global to common ground direction)

													if (q==0) assign_microstructure(cell, celldata,
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
													omatfile << local_quadrature_points_history[q].qpid << " " << local_quadrature_points_history[q].mat << std::endl;
											}
									}

							// Creating list of cell id/material mapping
							MPI_Barrier(FE_communicator);
							if (this_FE_process == 0){
									std::ifstream infile;
									std::ofstream outfile;
									std::string iline;

									sprintf(filename, "%s/cell_id_mat.list", macrostatelocout.c_str());
									outfile.open (filename);
									for (unsigned int ip=0; ip<n_FE_processes; ip++){
											char local_filename[1024];
											sprintf(local_filename, "%s/cell_id_mat.%d.list", macrostatelocout.c_str(), ip);
											infile.open (local_filename);
											while (getline(infile, iline)) outfile << iline << std::endl;
											infile.close();
											remove(local_filename);
									}
									outfile.close();
							}
							MPI_Barrier(FE_communicator);
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

			template <int dim>
					void FEProblem<dim>::set_boundary_values()
					{
						std::map<types::global_dof_index,double> boundary_values;

		        // define accelerations of boundary verticies, problem specific
						// e.g. defines acceleration of loaded verticies and sets edges to 0
						boundary_values = problem_type->set_boundary_conditions(present_time);

						for (std::map<types::global_dof_index, double>::const_iterator
							p = boundary_values.begin();
							p != boundary_values.end(); ++p){
							incremental_velocity(p->first) = p->second;
				    }
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
		boundary_values = problem_type->boundary_conditions_to_zero();

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
		// Preparing requirements for strain update
		FEValues<dim> fe_values (fe, quadrature_formula,
				update_values | update_gradients);
		std::vector<std::vector<Tensor<1,dim> > >
		displacement_update_grads (quadrature_formula.size(),
				std::vector<Tensor<1,dim> >(dim));

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				SymmetricTensor<2,dim> newton_strain_tensor;

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

					// Add current strain to strain history
					if(newtonstep>0){
						local_quadrature_points_history[q].hist_strain.add_current_strain(
									local_quadrature_points_history[q].new_strain[0][0],
									local_quadrature_points_history[q].new_strain[1][1],
									local_quadrature_points_history[q].new_strain[2][2],
									local_quadrature_points_history[q].new_strain[0][1],
									local_quadrature_points_history[q].new_strain[0][2],
									local_quadrature_points_history[q].new_strain[1][2]);
						local_quadrature_points_history[q].hist_strain.set_ID_to_get_results_from(local_quadrature_points_history[q].qpid);
					}
				}
			}
	}




	// The necessity for update and the subsequent spline analysis should be conducted in another class, which would
	// be provided with the strain state of all the QPs at each time step.
	// Maybe worth doing that in the spline class
	template <int dim>
	void FEProblem<dim>::check_strain_quadrature_point_history()
	{
		if (newtonstep > 0) dcout << "        " << "...checking quadrature points requiring update based on current strain..." << std::endl;
		
		double min_qp_strain;
		min_qp_strain = input_config.get<double>("model precision.md.min quadrature strain norm");

		for (typename DoFHandler<dim>::active_cell_iterator
				cell = dof_handler.begin_active();
				cell != dof_handler.end(); ++cell)
			if (cell->is_locally_owned())
			{
				SymmetricTensor<2,dim> newton_strain_tensor;

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
					// Uncomment one on the 4 following "if" statement to derive stress tensor from MD for:
					//   (i) all cells,
					//  (ii) cells in given location,
					// (iii) cells based on their id
					if (activate_md_update
						// otherwise MD simulation unecessary, because no significant volume change and MD will fail
										&& local_quadrature_points_history[q].upd_strain.norm() >= min_qp_strain//> 1.0e-10
						)
					//if (activate_md_update && cell->barycenter()(1) <  3.0*tt && cell->barycenter()(0) <  1.10*(ww - aa) && cell->barycenter()(0) > 0.0*(ww - aa))
					/*if (activate_md_update && (cell->active_cell_index() == 2922 || cell->active_cell_index() == 2923
						|| cell->active_cell_index() == 2924 || cell->active_cell_index() == 2487
						|| cell->active_cell_index() == 2488 || cell->active_cell_index() == 2489))*/ // For debug...
					{
						local_quadrature_points_history[q].to_be_updated = true;
					}
					else{
						local_quadrature_points_history[q].to_be_updated = false;
					}
				}
			}
	}




	template <int dim>
	void FEProblem<dim>::spline_building()
	{
		dcout << "           " << "...building splines..." << std::endl;

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
					local_quadrature_points_history[q].hist_strain.splinify(num_spline_points);

				}
			}
	}




	template <int dim>
	void FEProblem<dim>::spline_comparison()
	{
		dcout << "           " << "...computing similarity of splines..." << std::endl;

		// Building vector of (updateable) histories of cells on rank
		std::vector<MatHistPredict::Strain6D*> histories;
		int flag1 = 0;
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
				flag1++; 
				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					if(local_quadrature_points_history[q].to_be_updated)
					{
						histories.push_back(&local_quadrature_points_history[q].hist_strain);
						//local_quadrature_points_history[q].hist_strain.print();
					}
				}
			}
		MPI_Barrier(FE_communicator);
		// Launch MPI communication to compare strain histories on this rank with histories on all other ranks (including this one).
		// Results will be stored in the Strain6D objects - a vector of all other similar strain histories (i.e. within the given threshold difference).
		MatHistPredict::compare_histories_with_all_ranks(histories, acceptable_diff_threshold, FE_communicator);

		for(uint32_t i=0; i < histories.size(); i++) {
			char outhistfname[1024];
			sprintf(outhistfname, "%s/last.%d.similar_hist", macrostatelocout.c_str(), histories[i]->get_ID());
			histories[i]->most_similar_histories_to_file(outhistfname);

			sprintf(outhistfname, "%s/last.%d.all_similar_hist", macrostatelocout.c_str(), histories[i]->get_ID());
			histories[i]->all_similar_histories_to_file(outhistfname);

			/*sprintf(outhistfname, "%s/%d.%d.all_similar_hist", macrostatelocout.c_str(), timestep_no, histories[i]->get_ID());
			histories[i]->all_similar_histories_to_file(outhistfname);*/
		}

		dcout << "           " << "...computing quadrature points reduced dependencies..." << std::endl;
		// Use networkx to coarsegrain the strain similarity graph, outputting the final list of cells to update using MD (jobs_to_run.csv),
		// and where to get the stress results for the cells to be updated (mapping.csv). Script must run on only one rank.
		MPI_Barrier(FE_communicator);
		if(this_FE_process == 0) {
			char command[1024];
			sprintf(command,
					"python3 ../spline/coarsegrain_dependency_network.py %s %s/mapping.csv %d",
					macrostatelocout.c_str(),
					macrostatelocout.c_str(),
					triangulation.n_active_cells()*quadrature_formula.size()
					);
			int ret = system(command);
			if (ret!=0){
				std::cerr << "Failed completing coarse-graining of the update list dependency!" << std::endl;
				exit(1);
			}
		}
		MPI_Barrier(FE_communicator);

		for(uint32_t i=0; i < histories.size(); i++) {
			char mappingfname[1024];
			sprintf(mappingfname, "%s/mapping.csv", macrostatelocout.c_str());
			histories[i]->read_coarsegrain_dependency_mapping(mappingfname);
		}
	}




	template <int dim>
	void FEProblem<dim>::history_analysis()
	{
		dcout << "        " << "...comparing strain history of quadrature points to be updated..." << std::endl;
			
		num_spline_points = input_config.get<int>("model precision.spline.points");
    min_num_steps_before_spline = input_config.get<int>("model precision.spline.min steps");
		acceptable_diff_threshold = input_config.get<double>("model precision.spline.diff threshold");

		// Fit spline to all histories, and determine similarity graph (over all ranks)
		if(timestep > min_num_steps_before_spline) {
			spline_building();
			spline_comparison();
		}
	}




	template <int dim>
	void FEProblem<dim>::write_md_updates_list(ScaleBridgingData &scale_bridging_data)
	{
		std::vector<int> qpupdates;
		std::vector<double> strains;

		std::vector<SymmetricTensor<2,dim> > update_strains;

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
					if(local_quadrature_points_history[q].to_be_updated
							&& local_quadrature_points_history[q].hist_strain.run_new_md())
					{
						// The cell will get its stress from MD, but should it run an MD simulation?
						if (true
							// in case of extreme straining with reaxff
							/*&& !(avg_new_stress_tensor.norm() < 1.0e8 && avg_new_strain_tensor.norm() > 3.0)*/
							){
							//std::cout << "           "
							//		<< " cell_id "<< cell->active_cell_index()
							//		<< " upd norm " << local_quadrature_points_history[q].upd_strain.norm()
							//		<< " total norm " << local_quadrature_points_history[q].new_strain.norm()
							//		<< " total stress norm " << local_quadrature_points_history[q].new_stress.norm()
							//		<< std::endl;

							QP qp; // Struct that holds information for md job

							// Write strains since last update in a file named ./macrostate_storage/last.cellid-qid.strain
							//char cell_id[1024]; sprintf(cell_id, "%d", local_quadrature_points_history[q].qpid);
							//char filename[1024];

							SymmetricTensor<2,dim> rot_avg_upd_strain_tensor;

							rot_avg_upd_strain_tensor =
										rotate_tensor(local_quadrature_points_history[q].upd_strain, local_quadrature_points_history[q].rotam);
												
							for (int i=0; i<6; i++){
								qp.update_strain[i] = rot_avg_upd_strain_tensor.access_raw_entry(i); 
							}
							qp.id = local_quadrature_points_history[q].qpid;
							qp.material = celldata.get_composition(cell->active_cell_index());
							scale_bridging_data.update_list.push_back(qp);
							//sprintf(filename, "%s/last.%s.upstrain", macrostatelocout.c_str(), cell_id);
							//write_tensor<dim>(filename, rot_avg_upd_strain_tensor);

							// qpupdates.push_back(local_quadrature_points_history[q].qpid); //MPI list of qps to update on this rank
							//std::cout<< "local qpid "<< local_quadrature_points_history[q].qpid << std::endl;
						}
					} 
			}
		// Gathering in a single file all the quadrature points to be updated...
		// Might be worth replacing indivual local file writings by a parallel vector of string
		// and globalizing this vector before this final writing step.
		gather_qp_update_list(scale_bridging_data);
		//std::vector<int> all_qpupdates;
		///all_qpupdates = gather_vector<int>(qpupdates);
		;
		/*for (int i=0; i < all_qpupdates.size(); i++){
			QP qp;
			qp.id = all_qpupdates[i];
			qp.material = celldata.get_composition(qp.id);
			scale_bridging_data.update_list.push_back(qp);		
		}*/
		
	}

	template <int dim>
	void FEProblem<dim>::gather_qp_update_list(ScaleBridgingData &scale_bridging_data)
	{
		scale_bridging_data.update_list = gather_vector<QP>(scale_bridging_data.update_list);
	}
			
	template <int dim>
	template <typename T>
	std::vector<T> FEProblem<dim>::gather_vector(std::vector<T> local_vector)
	{
		// Gather a variable length vector held on each rank into one vector on rank 0
		int elements_on_this_proc = local_vector.size();
		std::vector<int> elements_per_proc(n_FE_processes); // number of elements on each rank
		MPI_Gather(&elements_on_this_proc, 	//sendbuf
					 	1,														//sendcount
						MPI_INT,											//sendtype
						&elements_per_proc.front(),		//recvbuf
						1,														//rcvcount
						MPI_INT,											//recvtype
						0,
						FE_communicator);	
		
		uint32_t total_elements = 0; 
		for (uint32_t i = 0; i < n_FE_processes; i++)
		{
			total_elements += elements_per_proc[i];
		}
		
		// Displacement local vector in the main vector for use in MPI_Gatherv
		int *disps = new int[n_FE_processes];
		for (int i = 0; i < n_FE_processes; i++)
		{
		   disps[i] = (i > 0) ? (disps[i-1] + elements_per_proc[i-1]) : 0;
		}
		
		typename mpi_type; 
		if      (typeid(T) == typeid(int))    mpi_type = MPI_INT;
		else if (typeid(T) == typeid(double)) mpi_type = MPI_DOUBLE;
		else if (typeid(T) == typeid(QP))			mpi_type = MPI_QP;
		else {
			dcout<<"Type not implemented in gather_vector"<<std::endl;
			exit(1);
		}

		// Populate a list with all elements requested
		std::vector<T> gathered_vector(total_elements); // vector with all elements from all ranks
		MPI_Gatherv(&local_vector.front(),     // *sendbuf,
            local_vector.size(),       	// sendcount,
            mpi_type,										// sendtype,
  					&gathered_vector.front(),	    // *recvbuf,
  					&elements_per_proc.front(),	// *recvcounts[],
						disps,											// displs[],
            mpi_type, 										//recvtype,
						0,
						FE_communicator);
		/*		
		if (this_FE_process == 0){
			dcout << "GATHER VECTOR OUTPUT " << total_elements << " ";
			for (int i = 0; i < gathered_vector.size(); i++)
			{
				dcout << gathered_vector[i] << " " ;
			}
			dcout << std::endl;
		}*/

		return gathered_vector;
	}

	QP get_qp_with_id (int cell_id, ScaleBridgingData scale_bridging_data)
	{
		QP qp; 
		bool found = false;
		int n_qp = scale_bridging_data.update_list.size();

		for (int i=0; i<n_qp; i++){
			if (scale_bridging_data.update_list[i].id == cell_id){
				qp = scale_bridging_data.update_list[i];
				found = true;
				break;
			}
		}
		if (found == false){
			std::cout << "Error: No QP objecr with id "<< cell_id << std::endl;
			exit(1);
		}
		return qp;
	}

	template <int dim>
	void FEProblem<dim>::update_stress_quadrature_point_history(const Vector<double>& displacement_update,
																															ScaleBridgingData scale_bridging_data)
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
				cell != dof_handler.end(); ++cell){
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

				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
					char cell_id[1024]; sprintf(cell_id, "%d", local_quadrature_points_history[q].hist_strain.get_ID_to_update_from());
					int qp_id = local_quadrature_points_history[q].hist_strain.get_ID_to_update_from();
				
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
						//bool load_stress = true;

						/*SymmetricTensor<4,dim> loc_stiffness;
						sprintf(filename, "%s/last.%s.stiff", macrostatelocout.c_str(), cell_id);
						read_tensor<dim>(filename, loc_stiffness);*/

						QP qp;
						qp = get_qp_with_id(qp_id, scale_bridging_data);
						//sprintf(filename, "%s/last.%s.stress", macrostatelocout.c_str(), cell_id);
						//load_stress = read_tensor<dim>(filename, loc_stress);
			//std::cout << "Putting stress into quadrature_points_history" << std::endl;
      //for (int i=0; i<6; i++){
      //  std::cout << " " << qp.update_stress[i];
      //} std::cout << std::endl;
						SymmetricTensor<2,dim> loc_stress(qp.update_stress);

						// Rotate the output stress wrt the flake angles
						loc_stress = rotate_tensor(loc_stress, transpose(local_quadrature_points_history[q].rotam));

						if (approx_md_with_hookes_law == false){
							local_quadrature_points_history[q].new_stress = loc_stress;
						}
						else {
							local_quadrature_points_history[q].new_stress = loc_stress + local_quadrature_points_history[q].old_stress;
						}

						// Resetting the update strain tensor
						local_quadrature_points_history[q].upd_strain = 0;
					}
					else{
					// Tangent stiffness computation of the new stress tensor
						local_quadrature_points_history[q].new_stress +=
							local_quadrature_points_history[q].new_stiff*local_quadrature_points_history[q].newton_strain;
					}
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
				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

				for (unsigned int q=0; q<quadrature_formula.size(); ++q){
					char cell_id[1024]; sprintf(cell_id, "%d", local_quadrature_points_history[q].qpid);

					if(local_quadrature_points_history[q].to_be_updated
							&& local_quadrature_points_history[q].hist_strain.run_new_md()){
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
	std::vector< std::vector< Vector<double> > >
	FEProblem<dim>::compute_history_projection_from_qp_to_nodes (FE_DGQ<dim> &history_fe, DoFHandler<dim> &history_dof_handler, std::string stensor) const
	{
		std::vector< std::vector< Vector<double> > >
		             history_field (dim, std::vector< Vector<double> >(dim)),
		             local_history_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
		             local_history_fe_values (dim, std::vector< Vector<double> >(dim));
		for (unsigned int i=0; i<dim; i++)
		  for (unsigned int j=0; j<dim; j++)
		  {
		    history_field[i][j].reinit(history_dof_handler.n_dofs());
		    local_history_values_at_qpoints[i][j].reinit(quadrature_formula.size());
		    local_history_fe_values[i][j].reinit(history_fe.dofs_per_cell);
		  }
		FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
		                                         quadrature_formula.size());
		FETools::compute_projection_from_quadrature_points_matrix
		          (history_fe,
		           quadrature_formula, quadrature_formula,
		           qpoint_to_dof_matrix);
		typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
		                                               endc = dof_handler.end(),
		                                               dg_cell = history_dof_handler.begin_active();
		for (; cell!=endc; ++cell, ++dg_cell){
			if (cell->is_locally_owned()){
				PointHistory<dim> *local_quadrature_points_history
				= reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert (local_quadrature_points_history >=
						&quadrature_point_history.front(),
						ExcInternalError());
				Assert (local_quadrature_points_history <
						&quadrature_point_history.back(),
						ExcInternalError());
				for (unsigned int i=0; i<dim; i++){
					for (unsigned int j=0; j<dim; j++)
					{
						for (unsigned int q=0; q<quadrature_formula.size(); ++q){
							if (stensor == "strain"){
								local_history_values_at_qpoints[i][j](q)
				                		   = local_quadrature_points_history[q].new_strain[i][j];
							}
							else if(stensor == "stress"){
								local_history_values_at_qpoints[i][j](q)
				                		   = local_quadrature_points_history[q].new_stress[i][j];
							}
							else{
								std::cerr << "Error: Neither 'stress' nor 'strain' to be projected to DOFs..." << std::endl;
							}
						}
						qpoint_to_dof_matrix.vmult (local_history_fe_values[i][j],
								local_history_values_at_qpoints[i][j]);
						dg_cell->set_dof_values (local_history_fe_values[i][j],
								history_field[i][j]);
					}
				}
			}
			else{
				for (unsigned int i=0; i<dim; i++){
					for (unsigned int j=0; j<dim; j++)
					{
						for (unsigned int q=0; q<quadrature_formula.size(); ++q){
							local_history_values_at_qpoints[i][j](q) = -1e+20;
						}
						qpoint_to_dof_matrix.vmult (local_history_fe_values[i][j],
								local_history_values_at_qpoints[i][j]);
						dg_cell->set_dof_values (local_history_fe_values[i][j],
								history_field[i][j]);
					}
				}
			}
		}

		return history_field;
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
			lhprocout << "timestep,time,qpid,cell,qpoint,material";
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
							<< "," << present_time
							<< "," << local_qp_hist[q].qpid
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
	void FEProblem<dim>::output_visualisation_history ()
	{
		// Data structure for VTK output
		DataOut<dim> data_out;
		data_out.attach_dof_handler (history_dof_handler);

		// Output of the cell norm of the averaged strain tensor over quadrature
		// points as a scalar
		std::vector<std::string> stensor_proj;
		stensor_proj.push_back("strain");
		stensor_proj.push_back("stress");

		std::vector<std::vector< std::vector< Vector<double> > > > tensor_proj;

		for(unsigned int i=0; i<stensor_proj.size(); i++){
			tensor_proj.push_back(compute_history_projection_from_qp_to_nodes (history_fe, history_dof_handler, stensor_proj[i]));
			data_out.add_data_vector (tensor_proj[i][0][0], stensor_proj[i]+"_xx");
			data_out.add_data_vector (tensor_proj[i][1][1], stensor_proj[i]+"_yy");
			data_out.add_data_vector (tensor_proj[i][2][2], stensor_proj[i]+"_zz");
			data_out.add_data_vector (tensor_proj[i][0][1], stensor_proj[i]+"_xy");
			data_out.add_data_vector (tensor_proj[i][0][2], stensor_proj[i]+"_xz");
			data_out.add_data_vector (tensor_proj[i][1][2], stensor_proj[i]+"_yz");
		}

		data_out.build_patches ();

		// Grouping spatially partitioned outputs
		std::string filename = macrologloc + "/" + "history-" + Utilities::int_to_string(timestep,4)
		+ "." + Utilities::int_to_string(this_FE_process,3)
		+ ".vtu";
		AssertThrow (n_FE_processes < 1000, ExcNotImplemented());

		std::ofstream output (filename.c_str());
		data_out.write_vtu (output);

		MPI_Barrier(FE_communicator); // just to be safe
		if (this_FE_process==0)
		{
			std::vector<std::string> filenames_loc;
			for (unsigned int i=0; i<n_FE_processes; ++i)
				filenames_loc.push_back ("history-" + Utilities::int_to_string(timestep,4)
			+ "." + Utilities::int_to_string(i,3)
			+ ".vtu");

			const std::string
			visit_master_filename = (macrologloc + "/" + "history-" +
					Utilities::int_to_string(timestep,4) +
					".visit");
			std::ofstream visit_master (visit_master_filename.c_str());
			data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
			//DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

			const std::string
			pvtu_master_filename = (macrologloc + "/" + "history-" +
					Utilities::int_to_string(timestep,4) +
					".pvtu");
			std::ofstream pvtu_master (pvtu_master_filename.c_str());
			data_out.write_pvtu_record (pvtu_master, filenames_loc);

			static std::vector<std::pair<double,std::string> > times_and_names;
			const std::string
						pvtu_master_filename_loc = ("history-" +
								Utilities::int_to_string(timestep,4) +
								".pvtu");
			times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename_loc));
			std::ofstream pvd_output (macrologloc + "/" + "history.pvd");
			data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
			//DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
		}
		MPI_Barrier(FE_communicator);
	}




	template <int dim>
	void FEProblem<dim>::output_visualisation_solution ()
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


		// Output of internal forces as a vector
		Vector<double> fint = compute_internal_forces ();
		std::vector<std::string>  fint_names (dim, "fint");
		data_out.add_data_vector (velocity,
				fint_names,
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

		MPI_Barrier(FE_communicator); // just to be safe
		if (this_FE_process==0)
		{
			std::vector<std::string> filenames_loc;
			for (unsigned int i=0; i<n_FE_processes; ++i)
				filenames_loc.push_back ("solution-" + Utilities::int_to_string(timestep,4)
			+ "." + Utilities::int_to_string(i,3)
			+ ".vtu");

			const std::string
			visit_master_filename = (macrologloc + "/" + "solution-" +
					Utilities::int_to_string(timestep,4) +
					".visit");
			std::ofstream visit_master (visit_master_filename.c_str());
			data_out.write_visit_record (visit_master, filenames_loc); // 8.4.1
			//DataOutBase::write_visit_record (visit_master, filenames_loc); // 8.5.0

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
			data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
			//DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
		}
		MPI_Barrier(FE_communicator);
	}



	template <int dim>
	void FEProblem<dim>::output_results ()
	{
		// Output local history by processor
		if(timestep%freq_output_lhist==0) output_lhistory ();

		// Output visualisation files for paraview
		if(timestep%freq_output_visu==0){
			output_visualisation_history();
			output_visualisation_solution();
		}
	}



	// Creation of a checkpoint with the bare minimum data to restart the simulation (i.e nodes information,
	// and quadrature point information)
	template <int dim>
	void FEProblem<dim>::checkpoint (char* timeid) const
	{
		char filename[1024];

		// Copy of the solution vector at the end of the presently converged time-step.
		MPI_Barrier(FE_communicator);
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
		MPI_Barrier(FE_communicator);

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
							   std::vector<std::string> mdt, Tensor<1,dim> cgd,
							   std::string twodmfile, double extrudel, int extrudep,
						     boost::property_tree::ptree inconfig, 
								 bool hookes_law){

		approx_md_with_hookes_law = hookes_law;
		input_config = inconfig;		
 
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

		// Setting up mesh
		twod_mesh_file = twodmfile;
		extrude_length = extrudel;
		extrude_points = extrudep;
		// Setting materials name list
		mdtype = mdt;

		// Setting up common ground direction for rotation from microstructure given orientation
		cg_dir = cgd;

		dcout << " Initiation of the Mesh...       " << std::endl;
		make_grid ();
		problem_type->define_boundary_conditions(&dof_handler);

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
		set_boundary_values();
	}



	template <int dim>
	void FEProblem<dim>::solve (int nstp, ScaleBridgingData &scale_bridging_data){

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

		check_strain_quadrature_point_history();
		history_analysis();

		MPI_Barrier(FE_communicator);
		write_md_updates_list(scale_bridging_data);

	}



	template <int dim>
	bool FEProblem<dim>::check (ScaleBridgingData scale_bridging_data){
		double previous_res;

		update_stress_quadrature_point_history (newton_update_displacement, scale_bridging_data);
					

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
			//write files here
			//char timeid[1024];
       }

        dcout << std::endl;
    }
}

#endif    
