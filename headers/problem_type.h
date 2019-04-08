#ifndef PROBLEM_TYPE_H
#define PROBLEM_TYPE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>
#include <tuple>

#include "boost/property_tree/ptree.hpp"

//#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include "fe-spline_problem_hopk.h"


namespace HMM {
	
	struct MeshDimensions
	{
		double x, y, z;
		uint32_t x_cells, y_cells, z_cells;
	};

  template <int dim>
  class ProblemType 
	{
    public:
			virtual void make_grid(parallel::shared::Triangulation<dim> &triangulation);
			virtual void define_boundary_values(typename DoFHandler<dim> dof_handler);
			virtual void set_boundary_values(typename DoFHandler<dim> dof_handler,
                              double fe_timestep_length,
                              double present_time,
                              Vector<double>              &incremental_velocity,
                              std::vector<bool>           &supp_boundary_dofs,
                              std::vector<bool>           &load_boundary_dofs
    );


			MeshDimensions read_mesh_dimensions(boost::property_tree::ptree input_config)
			{
				MeshDimensions mesh;	
				mesh.x = input_config.get<double>("continuum mesh.input.x length");
				mesh.y = input_config.get<double>("continuum mesh.input.y length");
				mesh.z = input_config.get<double>("continuum mesh.input.z length");
				mesh.x_cells = input_config.get<uint32_t>("continuum mesh.input.x cells");
				mesh.y_cells = input_config.get<uint32_t>("continuum mesh.input.y cells");
				mesh.z_cells = input_config.get<uint32_t>("continuum mesh.input.z cells");

				if (mesh.x < 0 || mesh.y < 0 || mesh.z < 0){
					fprintf(stderr, "Mesh lengths must be positive \n");
					exit(1);
				}
				if (mesh.x_cells < 1 || mesh.y_cells < 1 || mesh.z_cells < 1 ){
					fprintf(stderr, "Must be at least 1 cell per axis \n");
					exit(1);
				}
				return mesh;
			}
	};

	template <int dim>
	class DropWeight: public ProblemType<dim>
	{
		public:
			DropWeight (boost::property_tree::ptree input)
      {
        input_config = input;
      }

			void make_grid(parallel::shared::Triangulation<dim> &triangulation)
      {	
				MeshDimensions mesh = this->read_mesh_dimensions(input_config);
					
				// Generate grid centred on 0,0 ; the top face is in plane with z=0	
				Point<dim> corner1 (-mesh.x/2, -mesh.y/2, -mesh.z);
				Point<dim> corner2 (mesh.x/2, mesh.y/2, 0);

				std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells}; 
				GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
      }

			void define_boundary_values(typename DoFHandler<dim> dof_handler)
			{
				typename DoFHandler<dim>::active_cell_iterator cell;
        double eps = cell->minimum_vertex_distance();

				for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
					for (uint32_t face = 0; face < GeometryInfo<3>::faces_per_cell; ++face){
						for (uint32_t vert = 0; vert < GeometryInfo<3>::vertices_per_face; ++vert) {

							// Point coords
							double vertex_x = fabs(cell->face(face)->vertex(vert)(0));
							double vertex_y = fabs(cell->face(face)->vertex(vert)(1));

							// in plane distance between vertex and centre of dropweight, whihc is at 0,0
							double x_dist = (vertex_x - 0.);
							double y_dist = (vertex_y - 0.);
							double dcwght = sqrt( x_dist*x_dist + y_dist*y_dist );

							// is vertex impacted by the drop weight
							if ((dcwght < diam_weight/2.)){ 
								loaded_vertices.push_back( cell->face(face)->vertex_dof_index );
							}
													
							// is point on the edge, if so it will be kept stationary
							double delta = eps / 10.0; // in a grid, this will be small enough that only edges are used
							if (   vertex_x > ( mesh.x/2 - delta)  
							    || vertex_x < (-mesh.x/2 + delta) 
							    || vertex_y > ( mesh.y/2 - delta) 
							    || vertex_y < (-mesh.y/2 + delta))
							{
								support_vertices.push_back( cell->face(face)->vertex_dof_index );
							}
						}
					}
				}
			}

			// Might want to restructure this function to avoid repetitions
			// with boundary conditions correction performed at the end of the
			// assemble_system() function
			void set_boundary_values(DoFHandler<dim> &dof_handler,
															double fe_timestep_length,
															double present_time,
              								Vector<double>              &incremental_velocity,
															std::vector<bool>           &supp_boundary_dofs,
              								std::vector<bool>           &load_boundary_dofs
		)
			{
				double diam_weight = input_config.get<double>("drop weight.diameter");
				double tacc_vsupport = input_config.get<double>("drop weight.acceleration");
				double acc_steps = input_config.get<double>("drop weight.steps to accelerate");
			
				double accelerate_time;
				double travel_time;
				double decelerate_time;

				double inc_vsupport;

				// duration during which the boundary accelerates + delta for avoiding numerical error
				accelerate_time = acc_steps * fe_timestep_length + fe_timestep_length * 0.001;  
				travel_time = 0.0 * fe_timestep_length;
				decelerate_time = accelerate_time;
				bool is_loaded = true;

				//dcout << "Loading condition: " << std::flush;
				// acceleration of the loading support (reaching aimed velocity)
				if (present_time <= accelerate_time){
						//dcout << "ACCELERATE!!!" << std::flush;
						inc_vsupport = tacc_vsupport*fe_timestep_length;
				}
				// stationary motion of the loading support
				else if (present_time <= accelerate_time + travel_time){
						//dcout << "CRUISING!!!" << std::flush;
						inc_vsupport = 0.0;
				}
				// deccelaration of the loading support (return to 0 velocity)
				else if (present_time <= accelerate_time + travel_time + decelerate_time){
						//dcout << "DECCELERATE!!!" << std::flush;
						inc_vsupport = -1.0*tacc_vsupport*fe_timestep_length;
						is_loaded = false;
				}
				// stationary motion of the loading support
				else{
						//dcout << "NOT LOADED!!!" << std::flush;
						inc_vsupport = 0.0;
						is_loaded = false;
				}

				//dcout << " acceleration: " << tacc_vsupport << " - velocity increment: " << inc_vsupport << std::endl;

				FEValuesExtractors::Scalar x_component (dim-3);
				FEValuesExtractors::Scalar y_component (dim-2);
				FEValuesExtractors::Scalar z_component (dim-1);
				std::map<types::global_dof_index,double> boundary_values;

				supp_boundary_dofs.resize(dof_handler.n_dofs());
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
								load_boundary_dofs[cell->face(face)->vertex_dof_index (v, c)] = false;
							}
													
							// distance between q point and centre of dropweight, whihc is at 0,0,0
							double x_dist = (cell->face(face)->vertex(v)(0) - 0.);
							double y_dist = (cell->face(face)->vertex(v)(1) - 0.);
							double dcwght = sqrt( x_dist*x_dist + y_dist*y_dist );

							// is q point being impacted by the drop weight
							if(is_loaded){
								if ((dcwght < diam_weight/2.)){ 
									loaded_verticies
									load_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)] = true;
									boundary_values.insert(std::pair<types::global_dof_index, double>
									(cell->face(face)->vertex_dof_index (v, component), value));
								}
							}
													
							// Dimensions in x,y; assume grid centered on 0,0; see make_grid
							double x_length = input_config.get<double>(
														"continuum mesh.input.x length");
							double y_length = input_config.get<double>(
														"continuum mesh.input.y length");
							
							// Point coords
							double vertex_x = fabs(cell->face(face)->vertex(v)(0));
							double vertex_y = fabs(cell->face(face)->vertex(v)(1));

							// is point on the edge, if so it will be kept stationary
							if (   vertex_x > ( x_length/2 - 0.0000001)  
							    || vertex_x < (-x_length/2 + 0.0000001) 
							    || vertex_y > ( y_length/2 - 0.0000001) 
							    || vertex_y < (-y_length/2 + 0.0000001))
							{
								value = 0.;
								for (component = 0; component < 3; ++component)
								{
									supp_boundary_dofs[cell->face(face)->vertex_dof_index (v, component)] = true;
									boundary_values.insert(std::pair<types::global_dof_index, double>
									(cell->face(face)->vertex_dof_index (v, component), value));
								}
							}
						}
					}
				}

				for (std::map<types::global_dof_index, double>::const_iterator
					p = boundary_values.begin();
					p != boundary_values.end(); ++p){
						incremental_velocity(p->first) = p->second;
				}
			}


		private:
			boost::property_tree::ptree input_config;
			MeshDimensions							mesh;

			std::vector<int>			fixed_vertices;
      std::vector<int>			loaded_vertices;
	};
}

#endif
