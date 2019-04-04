#ifndef PROBLEM_TYPE_H
#define PROBLEM_TYPE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>

//#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>



namespace HMM {

  template <int dim>
  class ProblemType 
	{
    public:
      ProblemType(boost::property_tree::ptree input)
      {
				input_config = input;
      }

			parallel::shared::Triangulation<dim> make_grid()
			{
			}

			boost::property_tree::ptree input_config;
	};

	template <int dim>
	class ImpactTest: public ProblemType<dim>
	{
		public:
	
			parallel::shared::Triangulation<dim> make_grid()
      {
				double 		x_length = input_config.get<double>("continuum mesh.input.x length");
				double 		y_length = input_config.get<double>("continuum mesh.input.y length");
				double 		z_length = input_config.get<double>("continuum mesh.input.z length");
				uint32_t	x_cells = input_config.get<uint32_t>("continuum mesh.input.x cells");
				uint32_t	y_cells = input_config.get<uint32_t>("continuum mesh.input.y cells");
				uint32_t	z_cells = input_config.get<uint32_t>("continuum mesh.input.z cells");

				if (x_length < 0 || y_length < 0 || z_length < 0){
					fprintf(stderr, "Mesh lengths must be positive \n");
					exit(1);
				}
				if (x_cells < 1 || y_cells < 1 || z_cells < 1 ){
					fprintf(stderr, "Must be at least 1 cell per axis \n");
					exit(1);
				}
				
				// Generate grid centred on 0,0 ; the top face is in plane with z=0	
				Point<dim> corner1 (-x_length/2, -y_length/2, -z_length);
				Point<dim> corner2 (x_length/2, y_length/2, 0);

				std::vector<unsigned int> reps {x_cells, y_cells, z_cells}; 
        parallel::shared::Triangulation<dim> triangulation;

				GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);

				return triangulation;	
      }

	};
}

#endif
