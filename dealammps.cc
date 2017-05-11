/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "library.h"
#include "atom.h"

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

namespace micro
{
	using namespace LAMMPS_NS;

	// The initiation, namely the preparation of the data from which will
	// be ran the later tests at every quadrature point, should be ran on
	// as many processes as available, since it will be the only on going
	// task at the time it will be called.
	template <int dim>
	void
	lammps_initiation (std::vector<std::vector<double> >& voigt_stiffness)
	{
		int me;
		MPI_Comm_rank(MPI_COMM_WORLD,&me);

		LAMMPS *lmp = NULL;

		lmp = new LAMMPS(0,NULL,MPI_COMM_WORLD);

		char location[1024] = "../box";

		char infile[1024];
		char linec[1024];

		sprintf(linec, "variable locb string %s", location);
		lammps_command(lmp,linec);

		if (me == 0) std::cout << "   reading and executing in.init.lammps...       " << std::endl;
		sprintf(infile, "%s/%s", location, "in.init.lammps");
		lammps_file(lmp,infile);

		sprintf(linec, "write_restart %s", "PE_init_end.mstate");
		lammps_command(lmp,linec);

		/*sprintf(linec, "read_restart %s", "storage/PE_init_end.mstate");
		lammps_command(lmp,linec);*/

		sprintf(linec, "variable locbe string %s/%s", location, "ELASTIC");
		lammps_command(lmp,linec);
		// From a given state, use the 'in.stiffness.lammps' input file that computes
		// the 21 constants of the 6x6 symmetrical Voigt stiffness tensor.
		sprintf(infile, "%s/%s", location, "ELASTIC/in.elastic.lammps");
		lammps_file(lmp,infile);

		// Filling the 6x6 Voigt Sitffness tensor with its computed as variables
		// by LAMMPS
		if (me == 0) std::cout << "   retrieving stiffness tensor components...     " << std::endl;
		for(unsigned int k=0;k<2*dim;k++)
		   for(unsigned int l=k;l<2*dim;l++)
		   {
			   char vcoef[1024];
			   sprintf(vcoef, "C%d%dall", k+1, l+1);
			   voigt_stiffness[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;
			   if(k!=l) voigt_stiffness[l][k] = voigt_stiffness[k][l];
		   }

		// close down LAMMPS
		delete lmp;
	}


	// The local_testing function is ran on every quadrature point which
	// requires a stress_update. Since a quandrature point is only reached*
	// by a subset of processes N, we should automatically see lammps be
	// parallelized on the N processes.
	template <int dim>
	void
	lammps_local_testing (const std::vector<std::vector<double> >& strains,
						  std::vector<std::vector<double> >& stresses,
						  std::vector<std::vector<double> >& voigt_stiffness,
						  char* qptid,
						  MPI_Comm comm_lammps)
	{

		int me,nprocs;
		MPI_Comm_rank(comm_lammps,&me);
		MPI_Comm_size(comm_lammps,&nprocs);

		/*
		// Preparation of the local MPI communicators should be prepared in
		// advance...
		// Checking the validity of the amount of processes requested
		int nprocs_lammps = 1;
		if (nprocs_lammps > nprocs) {
			if (me == 0)
				printf("ERROR: LAMMPS cannot use more procs than available\n");
			MPI_Abort(MPI_COMM_WORLD,1);
		}

		// Selecting the nprocs_lammps first processes for this parallel
		// instanciation of lammps assigning color lammps=1 and comm_lammps
		// communicator
		int is_proc_concerned;
		if (me < nprocs_lammps) is_proc_concerned = 1;
		else is_proc_concerned = MPI_UNDEFINED;
		MPI_Comm comm_lammps;
		MPI_Comm_split(MPI_COMM_WORLD,is_proc_concerned,0,&comm_lammps);
		*/

		// Creating the corresponding lammps instantiation
		LAMMPS *lmp = NULL;

		int is_proc_concerned = 1;
		if (is_proc_concerned == 1){
			lmp = new LAMMPS(0,NULL,comm_lammps);

			char location[1024] = "../box";

			double dts = 2.0; // timestep length
			int nts = 1000; // number of timesteps
			char cmptid[1024] = "pr1"; // name of the stress compute to retrieve

			// Set initial state of the testing box (either from initial end state
			// or from previous testing end state).
			char linec[1024];
			//char indata[1024] = "PE_init_end.mstate";
			char indata[1024] = "PE_strain_end.mstate";

			sprintf(linec, "read_restart %s", indata);
			//sprintf(linit, "read_restart %s.%s", qptid, indata);
			lammps_command(lmp,linec);

			char infile[1024];
			sprintf(infile, "%s/%s", location, "in.strain.lammps");
			lammps_file(lmp,infile);

			// Implementation to be checked...
			int ncmds = 4;
			char **lines = new char*[ncmds];
			for(int i=0;i<ncmds;i++)  lines[i] = new char[1024];

			double pressure_scalar;
			double *stress_vector;

			sprintf(lines[0], "compute %s all pressure thermo_temp", cmptid);
			sprintf(lines[1],
					"fix 1 all deform 1  x erate %f  y erate %f  z erate %f"
					" xy erate %f xz erate %f yz erate %f"
					" remap x",
					strains[0][0]/(nts*dts),strains[1][1]/(nts*dts),strains[2][2]/(nts*dts),
					strains[0][1]/(nts*dts),strains[0][2]/(nts*dts),strains[1][2]/(nts*dts));
			sprintf(lines[2], "timestep %f", dts);
			sprintf(lines[3], "run %d", nts);
			//for(int i=0;i<ncmds;i++) printf("%s\n",lines[i]);

			// Apply a given strain on the box, and return to "equilibrium"
			lammps_commands_list(lmp,ncmds,lines);

			// Retieve the pressure and the stress computed using the compute 'cmptid'
			pressure_scalar = *((double *) lammps_extract_compute(lmp,cmptid,0,0));
			stress_vector = (double *) lammps_extract_compute(lmp,cmptid,0,1);

			// Convert vector to tensor (dimension independently...)
			for(unsigned int k=0;k<dim;k++) stresses[k][k] = stress_vector[k];
			for(unsigned int k=0;k<dim;k++)
			   for(unsigned int l=0;l<dim;l++)
				  if(k!=l) stresses[k][l] = stress_vector[k+l+2];

			// Save data to specific file for this quadrature point
			char outdata[1024] = "PE_strain_end.mstate";

			sprintf(linec, "write_restart %s", outdata);
			//sprintf(linec, "write_restart %s.%s", qptid, outdata);
			lammps_command(lmp,linec);

			//sprintf(linec, "read_restart %s", outdata);
			//sprintf(linec, "read_restart %s.%s", qptid, outdata);
			//lammps_command(lmp,linec);

			// From a given state, use the 'in.stiffness.lammps' input file that computes
			// the 21 constants of the 6x6 symmetrical Voigt stiffness tensor.
			sprintf(linec, "variable locbe string %s/%s", location, "ELASTIC");
			lammps_command(lmp,linec);
			sprintf(infile, "%s/%s", location, "ELASTIC/in.elastic.lammps");
			lammps_file(lmp,infile);

			// Filling the 6x6 Voigt Sitffness tensor with its computed as variables
			// by LAMMPS
			if (me == 0) std::cout << "   retrieving stiffness tensor components...     " << std::endl;
			for(unsigned int k=0;k<2*dim;k++)
			   for(unsigned int l=k;l<2*dim;l++)
			   {
				   char vcoef[1024];
				   sprintf(vcoef, "C%d%dall", k+1, l+1);
				   voigt_stiffness[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;
				   if(k!=l) voigt_stiffness[l][k] = voigt_stiffness[k][l];
			   }
		}

		// close down LAMMPS
		delete lmp;

		// close down the specific MPI communicator
		//if (is_proc_concerned == 1) MPI_Comm_free(&comm_lammps);
	}
}




namespace macro
{
  using namespace dealii;

  template <int dim>
  struct PointHistory
  {
    SymmetricTensor<2,dim> old_stress;
    SymmetricTensor<2,dim> new_stress;
    SymmetricTensor<2,dim> old_strain;
    SymmetricTensor<2,dim> new_strain;
    int nid;
  };

  template <int dim>
  inline
  SymmetricTensor<2,dim>
  get_strain (const FEValues<dim> &fe_values,
              const unsigned int   shape_func,
              const unsigned int   q_point)
  {
    SymmetricTensor<2,dim> tmp;

    for (unsigned int i=0; i<dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component (shape_func,q_point,i)[i];

    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=i+1; j<dim; ++j)
        tmp[i][j]
          = (fe_values.shape_grad_component (shape_func,q_point,i)[j] +
             fe_values.shape_grad_component (shape_func,q_point,j)[i]) / 2;

    return tmp;
  }

  template <int dim>
  inline
  SymmetricTensor<2,dim>
  get_strain (const std::vector<Tensor<1,dim> > &grad)
  {
    Assert (grad.size() == dim, ExcInternalError());

    SymmetricTensor<2,dim> strain;
    for (unsigned int i=0; i<dim; ++i)
      strain[i][i] = grad[i][i];

    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=i+1; j<dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
  }


  Tensor<2,2>
  get_rotation_matrix (const std::vector<Tensor<1,2> > &grad_u)
  {
    const double curl = (grad_u[1][0] - grad_u[0][1]);

    const double angle = std::atan (curl);

    const double t[2][2] = {{ cos(angle), sin(angle) },
      {-sin(angle), cos(angle) }
    };
    return Tensor<2,2>(t);
  }


  Tensor<2,3>
  get_rotation_matrix (const std::vector<Tensor<1,3> > &grad_u)
  {
    const Point<3> curl (grad_u[2][1] - grad_u[1][2],
                         grad_u[0][2] - grad_u[2][0],
                         grad_u[1][0] - grad_u[0][1]);

    const double tan_angle = std::sqrt(curl*curl);
    const double angle = std::atan (tan_angle);

    if (angle < 1e-9)
      {
        static const double rotation[3][3]
        = {{ 1, 0, 0}, { 0, 1, 0 }, { 0, 0, 1 } };
        static const Tensor<2,3> rot(rotation);
        return rot;
      }

    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1-c;

    const Point<3> axis = curl/tan_angle;
    const double rotation[3][3]
    = {{
        t *axis[0] *axis[0]+c,
        t *axis[0] *axis[1]+s *axis[2],
        t *axis[0] *axis[2]-s *axis[1]
      },
      {
        t *axis[0] *axis[1]-s *axis[2],
        t *axis[1] *axis[1]+c,
        t *axis[1] *axis[2]+s *axis[0]
      },
      {
        t *axis[0] *axis[2]+s *axis[1],
        t *axis[1] *axis[1]-s *axis[0],
        t *axis[2] *axis[2]+c
      }
    };
    return Tensor<2,3>(rotation);
  }




  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem ();
    ~ElasticProblem ();
    void run ();

  private:
    void make_grid ();
    void setup_system ();
    void do_timestep ();
    void set_boundary_values ();
    void assemble_system ();
    void solve_timestep ();
    unsigned int solve_linear_problem ();
    void error_estimation ();
    double determine_step_length () const;
    void move_mesh ();

    void setup_quadrature_point_history ();

    void update_quadrature_point_history (const Vector<double>& displacement_update);

    void output_results () const;

    double compute_residual () const;

    parallel::shared::Triangulation<dim> triangulation;
    DoFHandler<dim>      				dof_handler;
    FESystem<dim>        				fe;
    ConstraintMatrix     				hanging_node_constraints;

    const QGauss<dim>   				quadrature_formula;
    std::vector<PointHistory<dim> > 	quadrature_point_history;

    PETScWrappers::MPI::SparseMatrix	system_matrix;
    PETScWrappers::MPI::Vector      	system_rhs;

    Vector<double> 		     			newton_update;
    Vector<double> 		     			incremental_displacement;
    Vector<double> 		     			solution;

    Vector<float> 						error_per_cell;

    double              				present_time;
    double              				present_timestep;
    double              				end_time;
    unsigned int        				timestep_no;

    MPI_Comm 							mpi_communicator;
    const unsigned int 					n_mpi_processes;
    const unsigned int 					this_mpi_process;
    ConditionalOStream 					pcout;

    std::vector<types::global_dof_index> local_dofs_per_process;
    IndexSet 							locally_owned_dofs;
    IndexSet 							locally_relevant_dofs;
    unsigned int 						n_local_cells;

    // This will not be so constant anymore in our HMM: position dependent,
    // history dependent, and computed in an specific function using LAMMPS
    SymmetricTensor<4,dim> 				initial_stress_strain_tensor,
						   	   	   	   	stress_strain_tensor;
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
    const double rho = 7700;

    values = 0;
    values(dim-1) = -rho * g * 00000.;
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
  class IncrementalBoundaryValues :  public Function<dim>
  {
  public:
    IncrementalBoundaryValues (const double present_time,
                               const double present_timestep);

    virtual
    void
    vector_value (const Point<dim> &p,
                  Vector<double>   &values) const;

    virtual
    void
    vector_value_list (const std::vector<Point<dim> > &points,
                       std::vector<Vector<double> >   &value_list) const;

  private:
    const double velocity;
    const double present_time;
    const double present_timestep;
  };


  template <int dim>
  IncrementalBoundaryValues<dim>::
  IncrementalBoundaryValues (const double present_time,
                             const double present_timestep)
    :
    Function<dim> (dim),
    velocity (.001),
    present_time (present_time),
    present_timestep (present_timestep)
  {}


  template <int dim>
  void
  IncrementalBoundaryValues<dim>::
  vector_value (const Point<dim> &/*p*/,
                Vector<double>   &values) const
  {
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

    // All parts of the vector values are initiated to the given scalar.
    values = present_timestep * velocity;
  }



  template <int dim>
  void
  IncrementalBoundaryValues<dim>::
  vector_value_list (const std::vector<Point<dim> > &points,
                     std::vector<Vector<double> >   &value_list) const
  {
    const unsigned int n_points = points.size();

    Assert (value_list.size() == n_points,
            ExcDimensionMismatch (value_list.size(), n_points));

    for (unsigned int p=0; p<n_points; ++p)
      IncrementalBoundaryValues<dim>::vector_value (points[p],
                                                    value_list[p]);
  }


  // In order to modify the processes used by the deal.ii run, another
  // communicator should be used (e.g split from MPI_COMM_WORLD) and a
  // specific tag/color should be associated to the list of used processes
  // such as done in lammps_functions...
  template <int dim>
  ElasticProblem<dim>::ElasticProblem ()
    :
	  triangulation(mpi_communicator/*or MPI_COMM_WORLD*/),
	  dof_handler (triangulation),
	  fe (FE_Q<dim>(1), dim),
	  quadrature_formula (2),
  	  mpi_communicator (MPI_COMM_WORLD),
	  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
	  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
	  pcout (std::cout,(this_mpi_process == 0))
  {}



  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }



  template <int dim>
  void ElasticProblem<dim>::setup_quadrature_point_history ()
  {
	unsigned int our_cells = 0;
    for (typename Triangulation<dim>::active_cell_iterator
		 cell = triangulation.begin_active();
		 cell != triangulation.end(); ++cell)
    	if (cell->is_locally_owned()) ++our_cells;

	triangulation.clear_user_data();

    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (our_cells *
                                     quadrature_formula.size());

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
  }



  template <int dim>
  void ElasticProblem<dim>::set_boundary_values ()
  {
    FEValuesExtractors::Scalar t_component (dim-3);
    FEValuesExtractors::Scalar h_component (dim-2);
    FEValuesExtractors::Scalar v_component (dim-1);
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 12,
								 ZeroFunction<dim>(dim),
                                 boundary_values);

    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 22,
                                 IncrementalBoundaryValues<dim>(present_time,
                                                                present_timestep),
                                 boundary_values,
                                 fe.component_mask(h_component));

    for (std::map<types::global_dof_index, double>::const_iterator
       p = boundary_values.begin();
       p != boundary_values.end(); ++p)
         incremental_displacement(p->first) = p->second;
  }



  template <int dim>
  void ElasticProblem<dim>::assemble_system ()
  {
    system_rhs = 0;
    system_matrix = 0;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    BodyForce<dim>      body_force;
    std::vector<Vector<double> > body_force_values (n_q_points,
                                                    Vector<double>(dim));

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);

          const PointHistory<dim> *local_quadrature_points_data
            = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              for (unsigned int q_point=0; q_point<n_q_points;
                   ++q_point)
                {
                  const SymmetricTensor<2,dim>
                  eps_phi_i = get_strain (fe_values, i, q_point),
                  eps_phi_j = get_strain (fe_values, j, q_point);

                  cell_matrix(i,j)
                  += (eps_phi_i * initial_stress_strain_tensor * eps_phi_j
                      *
                      fe_values.JxW (q_point));
                }

          body_force.vector_value_list (fe_values.get_quadrature_points(),
                                        body_force_values);

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;

              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  const SymmetricTensor<2,dim> &new_stress
                    = local_quadrature_points_data[q_point].new_stress;

                  cell_rhs(i) += (body_force_values[q_point](component_i) *
                                  fe_values.shape_value (i,q_point)
                                  -
                                  new_stress *
                                  get_strain (fe_values,i,q_point))
                                 *
                                 fe_values.JxW (q_point);
                }
            }

          cell->get_dof_indices (local_dof_indices);

          hanging_node_constraints
          .distribute_local_to_global(cell_matrix, cell_rhs,
                                       local_dof_indices,
                                       system_matrix, system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    FEValuesExtractors::Scalar t_component (dim-3);
    FEValuesExtractors::Scalar h_component (dim-2);
    FEValuesExtractors::Scalar v_component (dim-1);
    std::map<types::global_dof_index,double> boundary_values;

    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 12,
								 ZeroFunction<dim>(dim),
                                 boundary_values);

    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 22,
								 ZeroFunction<dim>(dim),
                                 boundary_values,
                                 fe.component_mask(h_component));

    PETScWrappers::MPI::Vector tmp (locally_owned_dofs,mpi_communicator);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        tmp,
                                        system_rhs,
										false);
    newton_update = tmp;

  }



  template <int dim>
  unsigned int ElasticProblem<dim>::solve_linear_problem ()
  {
	PETScWrappers::MPI::Vector
	distributed_newton_update (locally_owned_dofs,mpi_communicator);
	distributed_newton_update = newton_update;

    SolverControl       solver_control (1000,
                                            1e-16*system_rhs.l2_norm());
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);

    // Apparently (according to step-17.tuto) the BlockJacobi preconditionner is
    // not optimal for large scale simulations.
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    cg.solve (system_matrix, distributed_newton_update, system_rhs,
              preconditioner);

    newton_update = distributed_newton_update;
    hanging_node_constraints.distribute (newton_update);

    const double alpha = determine_step_length();
    incremental_displacement.add (alpha, newton_update);

    return solver_control.last_step();
  }



  template <int dim>
  void ElasticProblem<dim>::update_quadrature_point_history
        (const Vector<double>& displacement_update)
  {
	std::vector<std::vector<double> >
		strain_vvector (dim, std::vector<double> (dim)),
	    stress_vvector (dim, std::vector<double> (dim)),
		local_voigt_stiffness (2*dim, std::vector<double> (2*dim));

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values | update_gradients);
    std::vector<std::vector<Tensor<1,dim> > >
    displacement_update_grads (quadrature_formula.size(),
                                  std::vector<Tensor<1,dim> >(dim));

    Assert (quadrature_point_history.size() > 0,
            ExcInternalError());

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
          fe_values.get_function_gradients (displacement_update,
                                            displacement_update_grads);

          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
        	  local_quadrature_points_history[q].old_strain =
        			  local_quadrature_points_history[q].new_strain;
        	  local_quadrature_points_history[q].new_strain +=
					  	  get_strain (displacement_update_grads[q]);

        	  local_quadrature_points_history[q].old_stress =
        			  local_quadrature_points_history[q].new_stress;

        	  // Conversion of the dealii::symmetrictensor into a std::vector<vector>
        	  // in order to pass it to the lammps function without using deal.ii depepent
        	  // types.
              for(unsigned int k=0;k<dim;k++)
                for(unsigned int l=0;l<dim;l++)
                  strain_vvector[k][l] = local_quadrature_points_history[q].new_strain[k][l];

            }
          }

	MPI_Barrier(mpi_communicator);

    // Regroup here all split strain_vvector in between processes in one main vvector

	// Each cell will be allocated NB processes, to create one instance of lammps for
	// each quad_point sequentially. Hopefully treating a new one each time one has been
	// treated. We should create a new flag and replace the "is_locally_owned". A cell
	// should be owned by NB processes.
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell){

		// Selecting the NB processes for this parallel instanciation of
		// lammps assigning color lammps=1 and comm_lammps communicator
		int lammps = MPI_UNDEFINED;
		/*if (cell->proc_locally_owned_by()) lammps = 1;
		else lammps = MPI_UNDEFINED;*/
		MPI_Comm comm_cell;
		MPI_Comm_split(MPI_COMM_WORLD,lammps,0,&comm_cell);

    	if (cell->is_locally_owned())
    	//if (cell->proc_locally_owned_by())
          {

          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
              // Rather than an int, build an ID as a unique string cell_num.quad_loc_num
              char *quad_id = new char[1024];
              sprintf(quad_id, "%d.%d", cell->index(), q);

        	  // Then the lammps function instanciates lammps, starting from an initial
        	  // microstructure and applying the complete new_strain or starting from
        	  // the microstructure at the old_strain and applying the difference between
        	  // the new_ and _old_strains, returns the new_stress state.
        	  micro::lammps_local_testing<dim> (strain_vvector,
												stress_vvector,
												local_voigt_stiffness,
												quad_id,
												comm_cell);

            }
          }
    }

	MPI_Barrier(MPI_COMM_WORLD);

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
          fe_values.get_function_gradients (displacement_update,
                                            displacement_update_grads);

          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
              // Convert 2*dim component vector new_stress to a SymmetricTensor<2,dim>
              for(unsigned int k=0;k<dim;k++)
                for(unsigned int l=0;l<dim;l++)
                	local_quadrature_points_history[q].new_stress[k][l] = stress_vvector[k][l];

              // Or maybe we should compute the new stiffness tensor and compute the new
              // stress using it instead of asking lammps to return the stress.
              // Check what kind of tensor we compute with lammps: linear, secant, or tangent?
              // Conversion to 3x3x3x3 stiffness tensor.
              //SymmetricTensor<4,dim> stress_strain_tensor = local_voigt_stiffness;
              /*const SymmetricTensor<2,dim> local_quadrature_points_history[q].new_strain
                = stress_strain_tensor * local_quadrature_points_history[q].new_strain;*/

        	  // Apply rotation of the sample to the new state tensors
              const Tensor<2,dim> rotation
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

              local_quadrature_points_history[q].new_stress
                = rotated_new_stress;
              local_quadrature_points_history[q].new_strain
                = rotated_new_strain;
            }
          }
	MPI_Barrier(MPI_COMM_WORLD);

  }


  template <int dim>
  double ElasticProblem<dim>::compute_residual () const
  {
	PETScWrappers::MPI::Vector residual
								(locally_owned_dofs, mpi_communicator);

    residual = 0;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    Vector<double>               cell_residual (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    BodyForce<dim>      body_force;
    std::vector<Vector<double> > body_force_values (n_q_points,
                                                    Vector<double>(dim));

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_residual = 0;
          fe_values.reinit (cell);

          const PointHistory<dim> *local_quadrature_points_data
            = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
          body_force.vector_value_list (fe_values.get_quadrature_points(),
                                        body_force_values);

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;

              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  const SymmetricTensor<2,dim> &old_stress
                    = local_quadrature_points_data[q_point].new_stress;

                  cell_residual(i) += (body_force_values[q_point](component_i) *
                                       fe_values.shape_value (i,q_point)
                                       -
                                       old_stress *
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

    // This manner to remove lines concerned with boundary conditions in the
    // residual vector does not yield the same norm for the residual vector
    // and the system_rhs vector (for which boundary conditions are applied
    // differently).
    // Is the value obtained with this method correct?
    // Should we proceed differently to obtain the same norm value? Although
    // localizing the vector (see step-17) does not change the norm value.
    std::vector<bool> boundary_dofs (dof_handler.n_dofs());
    DoFTools::extract_boundary_dofs (dof_handler,
                                     ComponentMask(),
                                     boundary_dofs);
    for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
      if (boundary_dofs[i] == true)
        residual(i) = 0;

    return residual.l2_norm();
  }



  template <int dim>
  void ElasticProblem<dim>::solve_timestep ()
  {
    double previous_res;

    do
      {
    	previous_res = compute_residual();
        pcout << "  Initial residual: "
                  << previous_res
                  << std::endl;

        for (unsigned int inner_iteration=0; inner_iteration<5; ++inner_iteration)
          {

            pcout << "    Assembling system..." << std::flush;
            assemble_system ();

            pcout << "    System - norm of rhs is " << system_rhs.l2_norm()
                  << std::endl;

            const unsigned int n_iterations = solve_linear_problem ();

            pcout << "    Solver - norm of newton update is " << newton_update.l2_norm()
                  << std::endl;
            pcout << "    Solver converged in " << n_iterations
                  << " iterations." << std::endl;

            pcout << "    Updating quadrature point data..." << std::flush;
            update_quadrature_point_history (newton_update);
            pcout << std::endl;

            previous_res = compute_residual();

            pcout << "  Residual: "
                      << previous_res
                      << std::endl
                      << "  -"
                      << std::endl;
          }
      } while (previous_res>1e-3);
  }




  template <int dim>
  void ElasticProblem<dim>::error_estimation ()
  {
	error_per_cell.reinit (triangulation.n_active_cells());
	KellyErrorEstimator<dim>::estimate (dof_handler,
	                                    QGauss<dim-1>(2),
	                                    typename FunctionMap<dim>::type(),
	                                    newton_update,
	                                    error_per_cell,
	                                    ComponentMask(),
	                                    0,
	                                    MultithreadInfo::n_threads(),
	                                    this_mpi_process);

	// Not too sure how is stored the vector 'distributed_error_per_cell',
	// it might be worth checking in case this is local, hence using a
	// lot of memory on a single process. This is ok, however it might
	// stupid to keep this vector global because the memory space will
	// be kept used during the whole simulation.
	const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells ();
	PETScWrappers::MPI::Vector
	distributed_error_per_cell (mpi_communicator,
	                            triangulation.n_active_cells(),
	                            n_local_cells);
	for (unsigned int i=0; i<error_per_cell.size(); ++i)
	  if (error_per_cell(i) != 0)
	    distributed_error_per_cell(i) = error_per_cell(i);
	distributed_error_per_cell.compress (VectorOperation::insert);

	error_per_cell = distributed_error_per_cell;
  }




  template <int dim>
  double ElasticProblem<dim>::determine_step_length() const
  {
    return 1.0;
  }




  template <int dim>
  void ElasticProblem<dim>::move_mesh ()
  {
    std::cout << "    Moving mesh..." << std::endl;

    std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                      false);
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active ();
         cell != dof_handler.end(); ++cell)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        if (vertex_touched[cell->vertex_index(v)] == false)
          {
            vertex_touched[cell->vertex_index(v)] = true;

            Point<dim> vertex_displacement;
            for (unsigned int d=0; d<dim; ++d)
              vertex_displacement[d]
                = incremental_displacement(cell->vertex_dof_index(v,d));

            cell->vertex(v) += vertex_displacement;
          }
  }



  template <int dim>
  void ElasticProblem<dim>::output_results () const
  {
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);

	std::vector<std::string>  solution_names (dim, "displacement");
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	   data_component_interpretation
	      (dim, DataComponentInterpretation::component_is_part_of_vector);
	data_out.add_data_vector (solution,
							  solution_names,
							  DataOut<dim>::type_dof_data,
							  data_component_interpretation);

	data_out.add_data_vector (error_per_cell, "error_per_cell");

	Vector<double> norm_of_strain (triangulation.n_active_cells());
	{
	  FEValues<dim> fe_values (fe, quadrature_formula,
							 update_values   | update_gradients |
							 update_quadrature_points | update_JxW_values);
	  std::vector<std::vector<Tensor<1,dim> > >
			   solution_grads (quadrature_formula.size(),
										  std::vector<Tensor<1,dim> >(dim));

	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  endc = dof_handler.end();
	  for (; cell!=endc; ++cell)
		  if (cell->is_locally_owned())
		  {
			  fe_values.reinit (cell);
			  fe_values.get_function_gradients (solution,
												solution_grads);
			  SymmetricTensor<2,dim> accumulated_strain;

			  for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				{
				  const SymmetricTensor<2,dim> new_strain
					 = get_strain (solution_grads[q]);
				  accumulated_strain += new_strain;
				  norm_of_strain(cell->active_cell_index())
					 = (accumulated_strain /
						quadrature_formula.size()).norm();
				}
		  }
		  else norm_of_strain(cell->active_cell_index()) = -1e+20;
	}
	data_out.add_data_vector (norm_of_strain, "norm_of_strain");

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
			  	  	  	  	  	  	     (cell->user_pointer())[q].old_stress;

			  norm_of_stress(cell->active_cell_index())
			  	  = (accumulated_stress / quadrature_formula.size()).norm();
		  }
		  else norm_of_stress(cell->active_cell_index()) = -1e+20;
	}
	data_out.add_data_vector (norm_of_stress, "norm_of_stress");

	std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());
	GridTools::get_subdomain_association (triangulation, partition_int);
	const Vector<double> partitioning(partition_int.begin(),
	                                  partition_int.end());
	data_out.add_data_vector (partitioning, "partitioning");

	data_out.build_patches ();

	std::string filename = "solution-" + Utilities::int_to_string(timestep_no,4)
						   + "." + Utilities::int_to_string(this_mpi_process,3)
						   + ".vtu";
	AssertThrow (n_mpi_processes < 1000, ExcNotImplemented());

	std::ofstream output (filename.c_str());
	data_out.write_vtu (output);

    if (this_mpi_process==0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<n_mpi_processes; ++i)
          filenames.push_back ("solution-" + Utilities::int_to_string(timestep_no,4)
                               + "." + Utilities::int_to_string(i,3)
                               + ".vtu");

        const std::string
        visit_master_filename = ("solution-" +
                                 Utilities::int_to_string(timestep_no,4) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        data_out.write_visit_record (visit_master, filenames);

        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string(timestep_no,4) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        static std::vector<std::pair<double,std::string> > times_and_names;
        times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename));
        std::ofstream pvd_output ("solution.pvd");
        data_out.write_pvd_record (pvd_output, times_and_names);
      }
  }




  template <int dim>
  void ElasticProblem<dim>::make_grid ()
  {
	std::vector< unsigned int > sizes (GeometryInfo<dim>::faces_per_cell);
	sizes[0] = 0; sizes[1] = 1;
	sizes[2] = 0; sizes[3] = 1;
	sizes[4] = 0; sizes[5] = 0;
	GridGenerator::hyper_cross(triangulation, sizes);
    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
             cell != triangulation.end();
                ++cell)
       for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
           {
              if (cell->face(f)->center()[0] == -0.5)
                 cell->face(f)->set_boundary_id (11);
              if (cell->face(f)->center()[0] == 1.5)
                 cell->face(f)->set_boundary_id (12);
              if (cell->face(f)->center()[1] == -0.5)
                 cell->face(f)->set_boundary_id (21);
              if (cell->face(f)->center()[1] == 1.5)
                 cell->face(f)->set_boundary_id (22);
              if (cell->face(f)->center()[2] == -0.5)
                 cell->face(f)->set_boundary_id (31);
              if (cell->face(f)->center()[2] == 0.5)
                 cell->face(f)->set_boundary_id (32);
           }
    triangulation.refine_global (3);
  }



  template <int dim>
  void ElasticProblem<dim>::setup_system ()
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
	                                              mpi_communicator,
	                                              locally_relevant_dofs);

	  system_matrix.reinit (locally_owned_dofs,
	                        locally_owned_dofs,
							sparsity_pattern,
	                        mpi_communicator);
	  system_rhs.reinit (locally_owned_dofs, mpi_communicator);

	  incremental_displacement.reinit (dof_handler.n_dofs());
	  newton_update.reinit (dof_handler.n_dofs());
	  solution.reinit (dof_handler.n_dofs());

	  setup_quadrature_point_history ();
  }



  template <int dim>
  void ElasticProblem<dim>::do_timestep ()
  {

    present_time += present_timestep;
    ++timestep_no;
    pcout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;
    if (present_time > end_time)
      {
        present_timestep -= (present_time - end_time);
        present_time = end_time;
      }

    incremental_displacement = 0;

    set_boundary_values ();

    update_quadrature_point_history (incremental_displacement);

    solve_timestep ();

    solution+=incremental_displacement;

    error_estimation ();

    output_results ();

    pcout << std::endl;
  }



  template <int dim>
  void ElasticProblem<dim>::run ()
  {
    // Since LAMMPS is highly scalable, the initiation number of processes NI
    // can basically be equal to the maximum number of available processes NT which
    // can directly be found in the MPI_COMM.
	pcout << " Initiation of LAMMPS Testing Box...       " << std::endl;
    std::vector<std::vector<double> >
    		initial_voigt_stiffness (2*dim, std::vector<double> (2*dim));

    micro::lammps_initiation<dim> (initial_voigt_stiffness);

    pcout << "    Voigt tensor:       " << std::endl;
    for (unsigned int k=0; k<2*dim; ++k){
    	for (unsigned int l=0; l<2*dim; ++l)
    		pcout << initial_voigt_stiffness[k][l] << "  ";
    	pcout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Conversion to 3x3x3x3 stiffness tensor.
    //initial_stress_strain_tensor = 0.0;

    present_time = 0;
    present_timestep = 1;
    end_time = 10;
    timestep_no = 0;

    make_grid ();

    pcout << "    Number of active cells:       "
          << triangulation.n_active_cells()
          << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (GridTools::
                count_cells_with_subdomain_association (triangulation,p));
    pcout << ")" << std::endl;

    setup_system ();

    pcout << "    Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (DoFTools::
                count_dofs_with_subdomain_association (dof_handler,p));
    pcout << ")" << std::endl;

    while (present_time < end_time)
      do_timestep ();
  }
}


// There are several number of processes encountered: (i) NT the highest provided
// as an argument to aprun, (ii) ND the number of processes provided to deal.ii
// [arbitrary], (iii) NI the number of processes provided to the lammps initiation
// [as close as possible to NT], and (iv) NB the number of processes provided to lammps
// testing [NT divided by the number of concurrent testing boxes].
int main (int argc, char **argv)
{
  try
    {
      using namespace macro;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Create a subset of MPI_WORLD_COMM for the reduced amount of processes
      // deal.ii will run on 'comm_dealii' known to that subset of processes only.

      // Split MPI_WORLD_COMM into NC comm_lammps of an equal amount of processes NB,
      // so that NB=NT/NC (thus NC%N=0) and min(|N-100|) for all NB.

      ElasticProblem<3> elastic_problem;
      elastic_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
