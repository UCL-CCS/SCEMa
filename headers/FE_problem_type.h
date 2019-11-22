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
			virtual void define_boundary_conditions(DoFHandler<dim> &dof_handler);
			virtual std::map<types::global_dof_index, double> set_boundary_conditions(uint32_t timestep, double dt);
			virtual std::map<types::global_dof_index, double> boundary_conditions_to_zero(uint32_t timestep);
                        virtual bool is_vertex_loaded(int index);

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

}

#endif
