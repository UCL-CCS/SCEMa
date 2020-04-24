#include "math_calc.h"

namespace HMM {

template <int dim>
class Dogbone: public ProblemType<dim>
{
public:
	Dogbone (boost::property_tree::ptree input)
{
		input_config = input;
		strain_rate = input_config.get<double>("problem type.strain rate");
}
	std::vector<double> mesh_manipulation_for_bc_application(parallel::shared::Triangulation<dim> &triangulation)
	{
		// finding longest dimension of the mesh
		std::vector<double> limits_x, limits_y, limits_z;
		limits_x = min_max_on_axis<dim>(triangulation, 0);
		limits_y = min_max_on_axis<dim>(triangulation, 1);
		limits_z = min_max_on_axis<dim>(triangulation, 2);

		/*std::cout << "x:" << limits_x[0] << "," << limits_x[1]
							  << " - y: " << limits_y[0] << "," << limits_y[1]
							  << " - z: " << limits_z[0] << "," << limits_z[1]
							  << std::endl;*/

		double len_x, len_y, len_z;
		len_x = limits_x[1] - limits_x[0];
		len_y = limits_y[1] - limits_y[0];
		len_z = limits_z[1] - limits_z[0];

		// rotating so that the longest axis is along the z axis
		double pipi = std::atan(1.0)*4;
		if (len_x > len_y && len_x > len_z){
			std::cout << "Rotating mesh axis X to Z" << std::endl;
			GridTools::rotate(pipi/2, 1, triangulation);
		}
		else if (len_y > len_x && len_y > len_z){
			std::cout << "Rotating mesh axis Y to Z" << std::endl;
			// This function requires a version deal.ii higher than 8.5.0
			GridTools::rotate(pipi/2, 0, triangulation);
		}
		else if (len_z >= len_y && len_z >= len_x){
			std::cout << "Longest axis is already aligned with Z-axis" << std::endl;
			// no rotation needed
		}

		// Shifting mesh so that the bottom points are in the x,y plane
		limits_z = min_max_on_axis<dim>(triangulation, 2);
		GridTools::shift(Point<3>(0, 0, -limits_z[0]), triangulation);

		// Checking that the bottom of the dogbone in the z-direction sits in the x-y plane
		limits_z = min_max_on_axis<dim>(triangulation, 2);
		Assert(limits_z[0] < 1.0e-16,
					ExcMessage("Error: the bottom of the dogbone in the z-direction sits in the x-y plane."));
		return limits_z;
	}

	void make_grid(parallel::shared::Triangulation<dim> &triangulation)
	{
		std::string mesh_input_style;
		mesh_input_style = input_config.get<std::string>("continuum mesh.input.style");

		if (mesh_input_style == "cuboid"){
			mesh = this->read_mesh_dimensions(input_config);

			// Generate block with bottom in plane 0,0. Strain applied in z axis
			Point<dim> corner1 (0,0,0);
			Point<dim> corner2 (mesh.x, mesh.y, mesh.z);

			std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells};
			GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
		}
		// does input styple contain "file"
		else if (mesh_input_style.find("file") != std::string::npos){
			this-> import_mesh(triangulation, input_config);

			// Repositioning the mesh automatically to apply the dogbone test specific
			// boundary conditions
			std::vector<double> limits_z = mesh_manipulation_for_bc_application(triangulation);

			// Storing the z dimension
			mesh.z = limits_z[1] - limits_z[0];
			//std::cout << "limits_z, and mesh.z" << limits_z[0] << " " << limits_z[1] << " " << mesh.z << std::endl;
		}
	}

	void define_boundary_conditions(DoFHandler<dim> &dof_handler)
	{
		typename DoFHandler<dim>::active_cell_iterator cell;

		for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
			double eps = cell->minimum_vertex_distance();
			double delta = eps / 10.0;
			for (uint32_t face = 0; face < GeometryInfo<3>::faces_per_cell; ++face){
				for (uint32_t vert = 0; vert < GeometryInfo<3>::vertices_per_face; ++vert) {

					// Point coords
					double vertex_z = cell->face(face)->vertex(vert)(2);

					// is vertex at base
					if ( abs(vertex_z-0.0) < delta ){
						for (uint32_t i=0; i<dim; i++){
							fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, i) );
						}
					}

					// is vertex on top
					if ( abs(vertex_z-mesh.z) < delta){
						// fix in x,y; load along z axis
						fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 0) );
						fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 1) );
						loaded_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 2) );
					}
				}
			}
		}
	}

	std::map<types::global_dof_index,double> set_boundary_conditions(uint32_t timestep, double dt)
									{
		// define accelerations of boundary verticies
		std::map<types::global_dof_index, double> boundary_values;
		types::global_dof_index vert;

		// fixed verticies have acceleration 0
		for (uint32_t i=0; i<fixed_vertices.size(); i++){
			vert = fixed_vertices[i];
			boundary_values.insert( std::pair<types::global_dof_index,double> (vert, 0.0) );
		}

		// apply constant strain to top
		// need to pass FE solver the velocity increment
		// first step
		double acceleration;
		if (timestep == 1){
			//acceleration = strain_rate * mesh.z / dt;
			acceleration = strain_rate * mesh.z / dt;
			//std::cout << "SET ACC " << acceleration << " "<< strain_rate * mesh.z << std::endl;
		}
		else {
			acceleration = 0;
		}
		/*if (timestep == 1){
						acceleration = mesh.z * strain_rate / dt;  
				}
				else {
						double current_time = timestep * dt;
						double current_length = mesh.z + mesh.z*(current_time * strain_rate);
						double current_velocity = strain_rate * current_length;

						double prev_time = (timestep - 1) * dt;
						double prev_length = mesh.z + mesh.z*(prev_time * strain_rate);
						double prev_velocity = strain_rate * prev_length;

						acceleration = (current_velocity - prev_velocity) / dt;
				}*/
		//std::cout << "ACCELERATION " << acceleration << std::endl;
		for (uint32_t i=0; i<loaded_vertices.size(); i++){
			vert = loaded_vertices[i];
			boundary_values.insert( std::pair<types::global_dof_index,double> (vert, acceleration) );
		}

		return boundary_values;
									}

	std::map<types::global_dof_index,double> boundary_conditions_to_zero(uint32_t timestep)
									{
		std::map<types::global_dof_index, double> boundary_values;
		uint32_t vert;

		for (uint32_t i=0; i<fixed_vertices.size(); i++){
			vert = fixed_vertices[i];
			boundary_values.insert( std::pair<types::global_dof_index,double> (vert, 0.0) );
		}

		for (uint32_t i=0; i<loaded_vertices.size(); i++){
			vert = loaded_vertices[i];
			boundary_values.insert( std::pair<types::global_dof_index,double> (vert, 0.0) );
		}

		return boundary_values;
									}

	bool is_vertex_loaded(int index)
	{
		bool vertex_loaded = false;
		if (std::find(loaded_vertices.begin(), loaded_vertices.end(), index) != loaded_vertices.end())
			vertex_loaded = true;

        return vertex_loaded;
	}



private:
	boost::property_tree::ptree input_config;
	MeshDimensions							mesh;

	std::vector<uint32_t>			fixed_vertices;
	std::vector<uint32_t>			loaded_vertices;

	double strain_rate;
};

}
