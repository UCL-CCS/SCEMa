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

	virtual void make_grid(parallel::shared::Triangulation<dim> &triangulation)=0;
	virtual void define_boundary_conditions(DoFHandler<dim> &dof_handler)=0;
	virtual std::map<types::global_dof_index, double> set_boundary_conditions(uint32_t timestep, double dt)=0;
	virtual std::map<types::global_dof_index, double> boundary_conditions_to_zero(uint32_t timestep)=0;
	virtual bool is_vertex_loaded(int index)=0;

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

	void import_mesh(parallel::shared::Triangulation<dim> &triangulation, boost::property_tree::ptree input_config)
	{
		std::string mesh_input_style = input_config.get<std::string>("continuum mesh.input.style");
		if (mesh_input_style == "file2D"){
			this-> import_2Dmesh(triangulation, input_config);
		}
		else if (mesh_input_style == "file3D"){
			this-> import_3Dmesh(triangulation, input_config);
		}
	}

	void import_2Dmesh(parallel::shared::Triangulation<dim> &triangulation, boost::property_tree::ptree input_config)
	{
		std::string filename = input_config.get<std::string>("continuum mesh.input.filename");
		uint32_t extrude_cells = input_config.get<uint32_t>("continuum mesh.input.extrude_cells");
		double extrude_length = input_config.get<double>("continuum mesh.input.extrude_length");

		std::ifstream iss(filename);
		if (iss.is_open()){
			//if(this_FE_process == 0) std::cout << "    Reading in 2D mesh " << std::endl;
			Triangulation<2> triangulation2D;
			GridIn<2> gridin;
			gridin.attach_triangulation(triangulation2D);
			gridin.read_msh(iss);

			GridGenerator::extrude_triangulation (triangulation2D, extrude_cells, extrude_length, triangulation);
		}
		else {
			fprintf(stderr, "Cannot find mesh file \n");
			exit(1);
		}
	}


	void import_3Dmesh(parallel::shared::Triangulation<dim> &triangulation, boost::property_tree::ptree input_config)
	{
		std::string filename = input_config.get<std::string>("continuum mesh.input.filename");

		std::ifstream iss(filename);
		if (iss.is_open()){
			//if(this_FE_process == 0) std::cout << "    Reading in 3D mesh " << std::endl;
			GridIn<3> gridin;
			gridin.attach_triangulation(triangulation);
			gridin.read_msh(iss);
		}
		else {
			fprintf(stderr, "Cannot find mesh file \n");
			exit(1);
		}
	}

};

}

#endif
