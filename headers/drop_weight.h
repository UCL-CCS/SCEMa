namespace HMM {

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
				mesh = this->read_mesh_dimensions(input_config);
					
				// Generate grid centred on 0,0 ; the top face is in plane with z=0	
				Point<dim> corner1 (-mesh.x/2, -mesh.y/2, -mesh.z);
				Point<dim> corner2 (mesh.x/2, mesh.y/2, 0);

				std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells}; 
				GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
      }

			void define_boundary_conditions(DoFHandler<dim> &dof_handler)
			{
				double diam_weight = input_config.get<double>("problem type.diameter");

				typename DoFHandler<dim>::active_cell_iterator cell;

				for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
        	double eps = cell->minimum_vertex_distance();
					for (uint32_t face = 0; face < GeometryInfo<3>::faces_per_cell; ++face){
						for (uint32_t vert = 0; vert < GeometryInfo<3>::vertices_per_face; ++vert) {

							// Point coords
							double vertex_x = cell->face(face)->vertex(vert)(0);
							double vertex_y = cell->face(face)->vertex(vert)(1);

							// in plane distance between vertex and centre of dropweight, whihc is at 0,0
							double x_dist = (vertex_x - 0.);
							double y_dist = (vertex_y - 0.);
							double dcwght = sqrt( x_dist*x_dist + y_dist*y_dist );

							// is vertex impacted by the drop weight
							if ((dcwght < diam_weight/2.)){ 
								//for (uint32_t i=0; i<dim; i++){
									loaded_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 2) );
								//}
							}
													
							// is point on the edge, if so it will be kept stationary
							double delta = eps / 10.0; // in a grid, this will be small enough that only edges are used
							if (   ( abs(vertex_x - mesh.x/2) < delta )  
							    || ( abs(vertex_x + mesh.x/2) < delta ) 
							    || ( abs(vertex_y - mesh.y/2) < delta ) 
							    || ( abs(vertex_y + mesh.y/2) < delta ))

							//if (   vertex_x > ( mesh.x/2 - delta)  
							//    || vertex_x < (-mesh.x/2 + delta) 
							//    || vertex_y > ( mesh.y/2 - delta) 
							//    || vertex_y < (-mesh.y/2 + delta))
							{
								for (uint32_t axis=0; axis<dim; axis++){
									fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, axis) );
								}
							}
						}
					}
				}
			}

			std::map<types::global_dof_index,double> set_boundary_conditions(double t)
			{
				// define accelerations of boundary verticies
				std::map<types::global_dof_index, double> boundary_values;
				types::global_dof_index vert;

				// fixed verticies have acceleration 0
				for (uint32_t i=0; i<fixed_vertices.size(); i++){
					vert = fixed_vertices[i];
					boundary_values.insert( std::pair<types::global_dof_index,double> (vert, 0.0) );
				}

				// loaded verticies have const acceleration for first acc_steps
				double acc_steps = input_config.get<double>("problem type.steps to accelerate");
				double acceleration = input_config.get<double>("problem type.acceleration");
      	double fe_timestep_length  = input_config.get<double>("continuum time.timestep length");
				double inc_acceleration = acceleration * fe_timestep_length;

				for (uint32_t i=0; i<loaded_vertices.size(); i++){
					if (t < acc_steps * fe_timestep_length){
	        	vert = loaded_vertices[i];
  	      	boundary_values.insert( std::pair<types::global_dof_index,double> (vert, inc_acceleration) );
					}
				}
			
				return boundary_values;
			}

			std::map<types::global_dof_index,double> boundary_conditions_to_zero()
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
	};

}
