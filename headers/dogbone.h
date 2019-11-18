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
			std::vector<bool>                 is_vertex_loaded;
			void make_grid(parallel::shared::Triangulation<dim> &triangulation)
      {	
				mesh = this->read_mesh_dimensions(input_config);
	
				// Generate block with bottom in plane 0,0. Strain applied in z axis	
				Point<dim> corner1 (0,0,0);
				Point<dim> corner2 (mesh.x, mesh.y, mesh.z);

				std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells}; 
				GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
      }

			void define_boundary_conditions(DoFHandler<dim> &dof_handler)
			{
				typename DoFHandler<dim>::active_cell_iterator cell;
				is_vertex_loaded.reserve(dof_handler.n_dofs());
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
								is_vertex_loaded[cell->face(face)->vertex_dof_index(vert, 2)] = true;
								//std::cout << "LOAD " << cell->face(face)->vertex_dof_index(vert, 2) << std::endl;
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


		private:
			boost::property_tree::ptree input_config;
			MeshDimensions							mesh;

			std::vector<uint32_t>			fixed_vertices;
      std::vector<uint32_t>			loaded_vertices;

			double strain_rate;
	};

}
