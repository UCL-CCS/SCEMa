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
#include <string>
#include <sys/stat.h>

#include <math.h>
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

namespace HMM
{
  using namespace dealii;
  using namespace LAMMPS_NS;

  template <int dim>
  struct PointHistory
  {
    SymmetricTensor<2,dim> old_stress;
    SymmetricTensor<2,dim> new_stress;
    SymmetricTensor<2,dim> old_strain;
    SymmetricTensor<2,dim> new_strain;
    SymmetricTensor<4,dim> old_stiff;
    SymmetricTensor<4,dim> new_stiff;
    int nid;
  };

  template <int dim>
  inline
  void
  read_tensor (char *filename, SymmetricTensor<2,dim> &tensor)
  {
    std::ifstream ifile;

    ifile.open (filename);
    if (ifile.is_open())
    {
    	for(unsigned int k=0;k<dim;k++)
    		for(unsigned int l=k;l<dim;l++)
    		{
				char line[1024];
				if(ifile.getline(line, sizeof(line)))
					tensor[k][l] = std::strtod(line, NULL);
    		}
    	ifile.close();
    }
	else std::cout << "Unable to open" << filename << " to read it" << std::endl;

  }

  template <int dim>
  inline
  void
  read_tensor (char *filename, SymmetricTensor<4,dim> &tensor)
  {
    std::ifstream ifile;

    ifile.open (filename);
    if (ifile.is_open())
    {
    	for(unsigned int k=0;k<dim;k++)
    		for(unsigned int l=k;l<dim;l++)
    			for(unsigned int m=0;m<dim;m++)
    				for(unsigned int n=m;n<dim;n++)
    				{
    					char line[1024];
    					if(ifile.getline(line, sizeof(line)))
    						tensor[k][l][m][n]= std::strtod(line, NULL);
    				}
    	ifile.close();
    }
	else std::cout << "Unable to open" << filename << " to read it" << std::endl;
  }

  template <int dim>
  inline
  void
  write_tensor (char *filename, SymmetricTensor<2,dim> &tensor)
  {
    std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				//std::cout << std::setprecision(16) << tensor[k][l] << std::endl;
				ofile << std::setprecision(16) << tensor[k][l] << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
  }

  template <int dim>
  inline
  void
  write_tensor (char *filename, SymmetricTensor<4,dim> &tensor)
  {
    std::ofstream ofile;

	ofile.open (filename);
	if (ofile.is_open())
	{
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				for(unsigned int m=0;m<dim;m++)
					for(unsigned int n=m;n<dim;n++)
						ofile << std::setprecision(16) << tensor[k][l][m][n] << std::endl;
		ofile.close();
	}
	else std::cout << "Unable to open" << filename << " to write in it" << std::endl;
  }

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


  // Computes the complete tanget elastic stiffness tensor and returns
  // a 3x3x3x3 SymmetricTensor.
  template <int dim>
  SymmetricTensor<4,dim>
  lammps_stiffness (void *lmp, char *location)
  {
	  int me;
	  MPI_Comm_rank(MPI_COMM_WORLD, &me);

	  SymmetricTensor<4,dim> initial_stress_strain_tensor;
	  SymmetricTensor<2,2*dim> tmp;

	  char cfile[1024];
	  char cline[1024];

	  sprintf(cline, "variable locbe string %s/%s", location, "ELASTIC");
	  lammps_command(lmp,cline);
	  // From a given state, use the 'in.stiffness.lammps' input file that computes
	  // the 21 constants of the 6x6 symmetrical Voigt stiffness tensor.
	  sprintf(cfile, "%s/%s", location, "ELASTIC/in.elastic.lammps");
	  lammps_file(lmp,cfile);

	  // Filling the 6x6 Voigt Sitffness tensor with its computed as variables
	  // by LAMMPS
	  for(unsigned int k=0;k<2*dim;k++)
		  for(unsigned int l=k;l<2*dim;l++)
		  {
			  char vcoef[1024];
			  sprintf(vcoef, "C%d%dall", k+1, l+1);
			  tmp[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;
		  }

	  // Write test... (on the data returned by lammps)

	  // Conversion of the 6x6 Voigt Stiffness Tensor into the 3x3x3x3
	  // Standard Stiffness Tensor
	  for(unsigned int i=0;i<2*dim;i++)
	  {
		  int k, l;
		  if     (i==3+0){k=1; l=2;}
		  else if(i==3+1){k=0; l=2;}
		  else if(i==3+2){k=0; l=1;}
		  else  /*(i<3)*/{k=i; l=i;}


		  for(unsigned int j=0;j<2*dim;j++)
		  {
			  int m, n;

			  if     (j==3+0){m=1; n=2;}
			  else if(j==3+1){m=0; n=2;}
			  else if(j==3+2){m=0; n=1;}
			  else  /*(j<3)*/{m=j; n=j;}

			  initial_stress_strain_tensor[k][l][m][n]=tmp[i][j];
			  /*initial_stress_strain_tensor[l][k][m][n]=tmp[i][j];
			  initial_stress_strain_tensor[k][l][n][m]=tmp[i][j];
			  initial_stress_strain_tensor[l][k][n][m]=tmp[i][j];*/

			  /*if(me==0) std::cout << k << l << m << n << " - "
					  	  	  	  << i << j << " - "
								  << initial_stress_strain_tensor[k][l][m][n]
							      << std::endl;*/
		  }
	  }

	  // Write test... (or actually just verify once that the conversion is accurate
	  // in 3D, and even 2D if possible)

	  /*// Symmetry checking
	  if(me==0) std::cout << initial_stress_strain_tensor[0][1][0][2] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[1][0][0][2] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[0][1][2][0] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[1][0][2][0] << std::endl;
	  if(me==0) std::cout << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[0][2][0][1] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[2][0][0][1] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[0][2][1][0] << std::endl;
	  if(me==0) std::cout << initial_stress_strain_tensor[2][0][1][0] << std::endl;*/

	  return initial_stress_strain_tensor;

  }


  // The initiation, namely the preparation of the data from which will
  // be ran the later tests at every quadrature point, should be ran on
  // as many processes as available, since it will be the only on going
  // task at the time it will be called.
  template <int dim>
  void
  lammps_initiation (SymmetricTensor<4,dim>& initial_stress_strain_tensor,
		  	  	  	 MPI_Comm comm_lammps)
  {
	  std::vector<std::vector<double> > tmp (2*dim, std::vector<double>(2*dim));

	  int me;
	  MPI_Comm_rank(comm_lammps, &me);

	  LAMMPS *lmp = NULL;

	  lmp = new LAMMPS(0,NULL,comm_lammps);

	  char location[1024] = "../box";
	  char outdata[1024] = "PE_init_end.mstate";

	  char storloc[1024] = "./nanostate_storage";
	  std::string sstorloc(storloc);
	  mkdir((sstorloc).c_str(), ACCESSPERMS);

	  char nanor[1024] = "./nanostate_output/";
	  std::string snanor(nanor);
	  mkdir((snanor).c_str(), ACCESSPERMS);

	  char nanorepo[1024];
	  sprintf(nanorepo, "%s%s", nanor, "init");
	  std::string snanorepo(nanorepo);
	  mkdir((snanorepo).c_str(), ACCESSPERMS);

	  char cfile[1024];
	  char cline[1024];

	  bool compute_equil = false;


	  // Passing location for input and output as variables
	  sprintf(cline, "variable locb string %s", location);
	  lammps_command(lmp,cline);
	  sprintf(cline, "variable loco string %s", nanorepo);
	  lammps_command(lmp,cline);

	  if (me == 0) std::cout << "   reading and executing in.set.lammps...       " << std::endl;
	  // Setting general parameters for LAMMPS independentely of what will be
	  // tested on the sample next.
	  sprintf(cfile, "%s/%s", location, "in.set.lammps");
	  lammps_file(lmp,cfile);

	  if (me == 0) std::cout << "   reading and executing in.init.lammps...       " << std::endl;
	  if (compute_equil)
	  {
		  // Compute initialization of the sample which minimizes the free energy,
		  // heat up and finally cool down the sample.
		  sprintf(cfile, "%s/%s", location, "in.init.lammps");
		  lammps_file(lmp,cfile);

		  sprintf(cline, "write_restart %s/%s", storloc, outdata);
		  lammps_command(lmp,cline);
	  }
	  else
	  {
		  // Reload from previously computed initial preparation (minimization and
		  // heatup/cooldown), this option shouldn't remain, as in the first step the
		  // preparation should always be computed.
		  sprintf(cline, "read_restart %s/%s", storloc, outdata);
		  lammps_command(lmp,cline);
	  }

	  // Compute the Tangent Stiffness Tensor at the initial state
	  initial_stress_strain_tensor = lammps_stiffness<dim>(lmp,location);

	  // close down LAMMPS
	  delete lmp;
  }


  // The local_testing function is ran on every quadrature point which
  // requires a stress_update. Since a quandrature point is only reached*
  // by a subset of processes N, we should automatically see lammps be
  // parallelized on the N processes.
  template <int dim>
  void
  lammps_local_testing (const SymmetricTensor<2,dim>& strains,
		  	  	  	  	SymmetricTensor<2,dim>& stresses,
						SymmetricTensor<4,dim>& initial_stress_strain_tensor,
						char* qptid,
						MPI_Comm comm_lammps)
  {
	  std::vector<std::vector<double> > tmp (2*dim, std::vector<double>(2*dim));

	  // Creating the corresponding lammps instantiation
	  LAMMPS *lmp = NULL;

	  lmp = new LAMMPS(0,NULL,comm_lammps);

	  int me;
	  MPI_Comm_rank(comm_lammps, &me);

	  char location[1024] = "../box";
	  char storloc[1024] = "./nanostate_storage";
	  std::string sstorloc(storloc);
	  mkdir((sstorloc).c_str(), ACCESSPERMS);

	  char nanor[1024] = "./nanostate_output/";
	  std::string snanor(nanor);
	  mkdir((snanor).c_str(), ACCESSPERMS);

	  char nanorepo[1024];
	  sprintf(nanorepo, "%s%s", nanor, qptid);
	  std::string snanorepo(nanorepo);
	  mkdir((snanorepo).c_str(), ACCESSPERMS);

	  bool compute_finit = true;
	  char initdata[1024];
	  sprintf(initdata, "%s", "PE_init_end.mstate");
	  char straindata[1024];
	  sprintf(straindata, "%s.%s", qptid, "PE_strain_end.mstate");

	  char cline[1024];
	  char cfile[1024];
	  char mfile[1024];

	  // Passing location for output as variable
	  sprintf(cline, "variable loco string %s", nanorepo);
	  lammps_command(lmp,cline);

	  // Setting general parameters for LAMMPS independentely of what will be
	  // tested on the sample next.
	  sprintf(cfile, "%s/%s", location, "in.set.lammps");
	  lammps_file(lmp,cfile);

	  // Set initial state of the testing box (either from initial end state
	  // or from previous testing end state).
	  if(compute_finit) sprintf(mfile, "%s/%s", storloc, initdata);
	  else sprintf(mfile, "%s/%s", storloc, straindata);

	  std::ifstream ifile(mfile);
	  if (!ifile.good()) std::cout << "Unable to open init_state file to read" << std::endl;

	  sprintf(cline, "read_restart %s", mfile); lammps_command(lmp,cline);

	  // Declaration of variables of in.strain.lammps
	  double dts = 2.0; // timestep length
	  sprintf(cline, "variable dts equal %f", dts); lammps_command(lmp,cline);

	  int nts = 200; // number of timesteps
	  sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

	  char cmptid[1024] = "pr1"; // name of the stress compute to retrieve
	  sprintf(cline, "variable cmptid string %s", cmptid); lammps_command(lmp,cline);

	  for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k;l<dim;l++)
		  {
			  sprintf(cline, "variable eeps_%d%d equal %f", k, l, strains[k][l]/(nts*dts));
			  lammps_command(lmp,cline);
		  }

	  // Run the NEMD simulations of the strained box
	  sprintf(cfile, "%s/%s", location, "in.strain.lammps");
	  lammps_file(lmp,cfile);

	  // Retieve the stress computed using the compute 'cmptid'
	  double *stress_vector;
	  stress_vector = (double *) lammps_extract_compute(lmp,cmptid,0,1);

	  // Convert vector to tensor (dimension independent fahsion...)
	  for(unsigned int k=0;k<dim;k++) stresses[k][k] = stress_vector[k];
	  for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k+1;l<dim;l++)
			  stresses[k][l] = stress_vector[k+l+2];

	  // Save data to specific file for this quadrature point
	  sprintf(cline, "write_restart %s/%s", storloc, straindata);
	  lammps_command(lmp,cline);

	  // Compute the Tangent Stiffness Tensor at the given stress/strain state
	  initial_stress_strain_tensor = lammps_stiffness<dim>(lmp, location);

	  // close down LAMMPS
	  delete lmp;
  }


  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem ();
    ~ElasticProblem ();
    void run ();

  private:
    void set_lammps_procs ();
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

    MPI_Comm 							dealii_communicator;
    const unsigned int 					n_dealii_processes;
    const unsigned int 					this_dealii_process;
    int 								dealii_pcolor;

    MPI_Comm 							lammps_global_communicator;
    MPI_Comm 							lammps_batch_communicator;
    int 								n_lammps_processes;
    int 								n_lammps_batch_processes;
    int 								n_lammps_batch;
    int 								this_lammps_process;
    int 								lammps_pcolor;

    ConditionalOStream 					pcout;

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
    unsigned int        				newtonstep_no;

    std::vector<types::global_dof_index> local_dofs_per_process;
    IndexSet 							locally_owned_dofs;
    IndexSet 							locally_relevant_dofs;
    unsigned int 						n_local_cells;

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
  // communicator should be used (e.g split from MPI_COMM_WORLD)
  template <int dim>
  ElasticProblem<dim>::ElasticProblem ()
    :
  	  dealii_communicator (MPI_COMM_WORLD),
	  n_dealii_processes (Utilities::MPI::n_mpi_processes(dealii_communicator)),
	  this_dealii_process (Utilities::MPI::this_mpi_process(dealii_communicator)),
	  dealii_pcolor (0),
	  pcout (std::cout,(this_dealii_process == 0)),
	  triangulation(dealii_communicator/*or MPI_COMM_WORLD*/),
	  dof_handler (triangulation),
	  fe (FE_Q<dim>(1), dim),
	  quadrature_formula (2)
  {}



  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }



  template <int dim>
  void ElasticProblem<dim>::setup_quadrature_point_history ()
  {
	triangulation.clear_user_data();
    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (n_local_cells *
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

    // History data at integration points initialization
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
    			local_quadrature_points_history[q].new_stiff = initial_stress_strain_tensor;
    			local_quadrature_points_history[q].new_stress = 0;
    		}
    	}

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
                  const SymmetricTensor<4,dim> &new_stiff
                    = local_quadrature_points_data[q_point].new_stiff;

                  const SymmetricTensor<2,dim>
                  eps_phi_i = get_strain (fe_values, i, q_point),
                  eps_phi_j = get_strain (fe_values, j, q_point);

                  cell_matrix(i,j)
                  += (eps_phi_i * new_stiff * eps_phi_j
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

    PETScWrappers::MPI::Vector tmp (locally_owned_dofs,dealii_communicator);
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
	distributed_newton_update (locally_owned_dofs,dealii_communicator);
	distributed_newton_update = newton_update;

    SolverControl       solver_control (1000,
                                            1e-16*system_rhs.l2_norm());
    PETScWrappers::SolverCG cg (solver_control,
                                dealii_communicator);

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
	char storloc[1024] = "./macrostate_storage";
	std::string macrorepo(storloc);
	mkdir((macrorepo).c_str(), ACCESSPERMS);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values | update_gradients);
    std::vector<std::vector<Tensor<1,dim> > >
    displacement_update_grads (quadrature_formula.size(),
                                  std::vector<Tensor<1,dim> >(dim));

    Assert (quadrature_point_history.size() > 0,
            ExcInternalError());

	int nqptbu = 0;

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

    			local_quadrature_points_history[q].old_stiff =
    					local_quadrature_points_history[q].new_stiff;

    			// Store strains in a file named ./macrostate_storage/time.it-cellid.qid.strain
				char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);
				char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
				char filename[1024];

				sprintf(filename, "%s/%s.%s.strain", storloc, time_id, quad_id);
				write_tensor<dim>(filename, local_quadrature_points_history[q].new_strain);

				// Check if this is a good position for setting criterion of elastic regime?
				// Or maybe a separate loop?
				// If parallel task, need to retrieve information...
    		}
    	}

	MPI_Barrier(dealii_communicator);

	// For Debug...
//    for (typename DoFHandler<dim>::active_cell_iterator
//    		cell = dof_handler.begin_active();
//    		cell != dof_handler.end(); ++cell)
//    {
//    	for (unsigned int q=0; q<quadrature_formula.size(); ++q)
//    	{
//    		//test_if q must be updated...
//    		int q_to_be_updated = 1;
//    		if (/*q_to_be_updated*/cell->active_cell_index() == 0)
//    		{
//    			nqptbu++;
//    			// check returned value...
//    			if (lammps_pcolor == (nqptbu%n_lammps_batch))
//    			{
//    				int me;
//					MPI_Comm_rank(lammps_communicator, &me);
//    				std::cout << "nqptbu: " << nqptbu
//						  << " - cell / qp : " << cell->active_cell_index() << "/" << q
//						  << " - proc_world_rank: " << this_lammps_process
//    					  << " - lammps batch computed: " << (nqptbu%n_lammps_batch)
//						  << " - lammps batch color: " << lammps_pcolor
//						  << " - proc_batch_rank: " << me
//						  << std::endl;
//    			}
//    		}
//    	}
//    }

	nqptbu = 0;
    for (typename DoFHandler<dim>::active_cell_iterator
    		cell = dof_handler.begin_active();
    		cell != dof_handler.end(); ++cell)
    {
    	for (unsigned int q=0; q<quadrature_formula.size(); ++q)
    	{
			SymmetricTensor<2,dim> loc_strain, loc_stress;
			SymmetricTensor<4,dim> loc_stiffness;

			// Restore the strain tensor from the file ./macrostate_storage/time.it-cellid.qid.strain
			char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);
			char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
			char filename[1024];

			sprintf(filename, "%s/%s.%s.strain", storloc, time_id, quad_id);
			read_tensor<dim>(filename, loc_strain);

    		//test_if q must be updated...
    		int q_to_be_updated = 0;
    		if (cell->active_cell_index() == 0 && q == 0) q_to_be_updated = 1;
    		if (q_to_be_updated)
    		{
    			nqptbu++;
    			if (lammps_pcolor == (nqptbu%n_lammps_batch))
    			{
    				int me;
					MPI_Comm_rank(lammps_batch_communicator, &me);
    				std::cout << "nqptbu: " << nqptbu
						  << " - cell / qp : " << cell->active_cell_index() << "/" << q
						  << " - proc_world_rank: " << this_lammps_process
    					  << " - lammps batch computed: " << (nqptbu%n_lammps_batch)
						  << " - lammps batch color: " << lammps_pcolor
						  << " - proc_batch_rank: " << me
						  << std::endl;

    				// Then the lammps function instanciates lammps, starting from an initial
    				// microstructure and applying the complete new_strain or starting from
    				// the microstructure at the old_strain and applying the difference between
    				// the new_ and _old_strains, returns the new_stress state.
    				lammps_local_testing<dim> (loc_strain,
    						loc_stress,
    						loc_stiffness,
    						quad_id,
							lammps_batch_communicator);

    			}
    		}
    		else
    		{
				// For debugg using a linear constitutive equation
				loc_stiffness = initial_stress_strain_tensor;
				loc_stress
					= loc_stiffness
					* loc_strain;
    		}

			// Write the new stress and stiffness tensors into two files, respectively
			// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
			sprintf(filename, "%s/%s.%s.stress", storloc, time_id, quad_id);
			write_tensor<dim>(filename, loc_stress);

			sprintf(filename, "%s/%s.%s.stiff", storloc, time_id, quad_id);
			write_tensor<dim>(filename, loc_stiffness);
    	}
    }

    MPI_Barrier(lammps_global_communicator);

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
    			// Restore the new stress and stiffness tensors from two files, respectively
				// ./macrostate_storage/time.it-cellid.qid.stress and ./macrostate_storage/time.it-cellid.qid.stiff
    			char time_id[1024]; sprintf(time_id, "%d-%d", timestep_no, newtonstep_no);
    			char quad_id[1024]; sprintf(quad_id, "%d-%d", cell->active_cell_index(), q);
    			char filename[1024];

    			sprintf(filename, "%s/%s.%s.stress", storloc, time_id, quad_id);
    			read_tensor<dim>(filename, local_quadrature_points_history[q].new_stress);

    			sprintf(filename, "%s/%s.%s.stiff", storloc, time_id, quad_id);
    			read_tensor<dim>(filename, local_quadrature_points_history[q].new_stiff);

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
  }



  template <int dim>
  double ElasticProblem<dim>::compute_residual () const
  {
	PETScWrappers::MPI::Vector residual
								(locally_owned_dofs, dealii_communicator);

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
            pcout << std::endl;
            update_quadrature_point_history (newton_update);
            pcout << std::endl;

            previous_res = compute_residual();

            pcout << "  Residual: "
                      << previous_res
                      << std::endl
                      << "  -"
                      << std::endl;

            ++newtonstep_no;
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
	                                    this_dealii_process);

	// Not too sure how is stored the vector 'distributed_error_per_cell',
	// it might be worth checking in case this is local, hence using a
	// lot of memory on a single process. This is ok, however it might
	// stupid to keep this vector global because the memory space will
	// be kept used during the whole simulation.
	const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells ();
	PETScWrappers::MPI::Vector
	distributed_error_per_cell (dealii_communicator,
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

	// Macroscale results output repository
	std::string macrorepo = "./macroscale_output/";
	mkdir((macrorepo).c_str(), ACCESSPERMS);

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

	std::string filename = macrorepo + "solution-" + Utilities::int_to_string(timestep_no,4)
						   + "." + Utilities::int_to_string(this_dealii_process,3)
						   + ".vtu";
	AssertThrow (n_dealii_processes < 1000, ExcNotImplemented());

	std::ofstream output (filename.c_str());
	data_out.write_vtu (output);

    if (this_dealii_process==0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<n_dealii_processes; ++i)
          filenames.push_back (macrorepo + "solution-" + Utilities::int_to_string(timestep_no,4)
                               + "." + Utilities::int_to_string(i,3)
                               + ".vtu");

        const std::string
        visit_master_filename = (macrorepo + "solution-" +
                                 Utilities::int_to_string(timestep_no,4) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        //data_out.write_visit_record (visit_master, filenames); // 8.4.1
        DataOutBase::write_visit_record (visit_master, filenames); // 8.5.0

        const std::string
        pvtu_master_filename = (macrorepo + "solution-" +
                                Utilities::int_to_string(timestep_no,4) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        static std::vector<std::pair<double,std::string> > times_and_names;
        times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename));
        std::ofstream pvd_output (macrorepo + "solution.pvd");
        //data_out.write_pvd_record (pvd_output, times_and_names); // 8.4.1
        DataOutBase::write_pvd_record (pvd_output, times_and_names); // 8.5.0
      }
  }



  // There are several number of processes encountered: (i) n_lammps_processes the highest provided
  // as an argument to aprun, (ii) ND the number of processes provided to deal.ii
  // [arbitrary], (iii) NI the number of processes provided to the lammps initiation
  // [as close as possible to n_lammps_processes], and (iv) n_lammps_batch_processes the number of processes provided to one lammps
  // testing [NT divided by n_lammps_batch the number of concurrent testing boxes].
  template <int dim>
  void ElasticProblem<dim>::set_lammps_procs ()
  {
	  // Create a communicator for all processes allocated to lammps
	  MPI_Comm_dup(MPI_COMM_WORLD, &lammps_global_communicator);

	  MPI_Comm_rank(lammps_global_communicator,&this_lammps_process);
	  MPI_Comm_size(lammps_global_communicator,&n_lammps_processes);

	  // Arbitrary setting of NB and NT
	  n_lammps_batch_processes = 2;
	  n_lammps_batch = int(n_lammps_processes/n_lammps_batch_processes);
	  if(n_lammps_batch == 0) {n_lammps_batch=1; n_lammps_batch_processes=n_lammps_processes;}

	  // LAMMPS processes color: regroup processes by batches of size NB, except
	  // the last ones (me >= NB*NC) to create batches of only NB processes, nor smaller.
	  lammps_pcolor = MPI_UNDEFINED;
	  if(this_lammps_process < n_lammps_batch_processes*n_lammps_batch)
		  lammps_pcolor = int(this_lammps_process/n_lammps_batch_processes);

	  // Definition of the communicators
	  MPI_Comm_split(lammps_global_communicator, lammps_pcolor, this_lammps_process, &lammps_batch_communicator);

	  // Recapitulating allocation of each process to deal and lammps
	  std::cout << "proc world rank: " << this_lammps_process
			    << " - deal color: " << dealii_pcolor
				<< " - lammps color: " << lammps_pcolor << std::endl;
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
    triangulation.refine_global (1);
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
	                                              dealii_communicator,
	                                              locally_relevant_dofs);

	  system_matrix.reinit (locally_owned_dofs,
	                        locally_owned_dofs,
							sparsity_pattern,
							dealii_communicator);
	  system_rhs.reinit (locally_owned_dofs, dealii_communicator);

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

    newtonstep_no = 0;

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
	// Define groups of processes from all the processes allocated to LAMMPS
	// simulations. Associated communicators are defined as well.
    set_lammps_procs();

	pcout << " Initiation of LAMMPS Testing Box...       " << std::endl;

    // Since LAMMPS is highly scalable, the initiation number of processes NI
    // can basically be equal to the maximum number of available processes NT which
    // can directly be found in the MPI_COMM.
    lammps_initiation<dim> (initial_stress_strain_tensor, lammps_global_communicator);

//    double mu = 9.695e10, lambda = 7.617e10;
//    for (unsigned int i=0; i<dim; ++i)
//      for (unsigned int j=0; j<dim; ++j)
//        for (unsigned int k=0; k<dim; ++k)
//          for (unsigned int l=0; l<dim; ++l)
//        	  initial_stress_strain_tensor[i][j][k][l]
//							= (((i==k) && (j==l) ? mu : 0.0) +
//                               ((i==l) && (j==k) ? mu : 0.0) +
//                               ((i==j) && (k==l) ? lambda : 0.0));

    present_time = 0;
    present_timestep = 1;
    end_time = 1;
    timestep_no = 0;

    make_grid ();

    pcout << "    Number of active cells:       "
          << triangulation.n_active_cells()
          << " (by partition:";
    for (unsigned int p=0; p<n_dealii_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (GridTools::
                count_cells_with_subdomain_association (triangulation,p));
    pcout << ")" << std::endl;

    setup_system ();

    pcout << "    Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << " (by partition:";
    for (unsigned int p=0; p<n_dealii_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (DoFTools::
                count_dofs_with_subdomain_association (dof_handler,p));
    pcout << ")" << std::endl;

    pcout << " Beginning of Time-Stepping...       " << std::endl;
    while (present_time < end_time)
      do_timestep ();
  }
}



int main (int argc, char **argv)
{
  try
    {
      using namespace HMM;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

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
