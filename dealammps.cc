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


// @sect3{Include files}

// As usual, the first few include files are already known, so we will not
// comment on them further.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// In this example, we need vector-valued finite elements. The support for
// these can be found in the following include file:
#include <deal.II/fe/fe_system.h>
// We will compose the vector-valued finite elements from regular Q1 elements
// which can be found here, as usual:
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>

// This again is C++:
#include <fstream>
#include <iostream>

// The last step is as in previous programs. In particular, just like in
// step-7, we pack everything that's specific to this program into a namespace
// of its own.
namespace Step8
{
  using namespace dealii;

  // @sect3{The <code>PointHistory</code> class}

  // As was mentioned in the introduction, we have to store the old stress in
  // quadrature point so that we can compute the residual forces at this point
  // during the next time step. This alone would not warrant a structure with
  // only one member, but in more complicated applications, we would have to
  // store more information in quadrature points as well, such as the history
  // variables of plasticity, etc. In essence, we have to store everything
  // that affects the present state of the material here, which in plasticity
  // is determined by the deformation history variables.
  //
  // We will not give this class any meaningful functionality beyond being
  // able to store data, i.e. there are no constructors, destructors, or other
  // member functions. In such cases of `dumb' classes, we usually opt to
  // declare them as <code>struct</code> rather than <code>class</code>, to
  // indicate that they are closer to C-style structures than C++-style
  // classes.
  template <int dim>
  struct PointHistory
  {
    SymmetricTensor<2,dim> old_stress;
    double old_norm;
  };

  // @sect3{The stress-strain tensor}

  // Next, we define the linear relationship between the stress and the strain
  // in elasticity. It is given by a tensor of rank 4 that is usually written
  // in the form $C_{ijkl} = \mu (\delta_{ik} \delta_{jl} + \delta_{il}
  // \delta_{jk}) + \lambda \delta_{ij} \delta_{kl}$. This tensor maps
  // symmetric tensor of rank 2 to symmetric tensors of rank 2. A function
  // implementing its creation for given values of the Lame constants $\lambda$
  // and $\mu$ is straightforward:
  template <int dim>
  SymmetricTensor<4,dim>
  get_stress_strain_tensor (const double lambda, const double mu)
  {
    SymmetricTensor<4,dim> tmp;
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        for (unsigned int k=0; k<dim; ++k)
          for (unsigned int l=0; l<dim; ++l)
            tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
                               ((i==l) && (j==k) ? mu : 0.0) +
                               ((i==j) && (k==l) ? lambda : 0.0));
    return tmp;
  }

  // @sect3{Auxiliary functions}

  // The first one computes the symmetric strain tensor for shape function
  // <code>shape_func</code> at quadrature point <code>q_point</code> by
  // forming the symmetric gradient of this shape function. We need that when
  // we want to form the matrix, for example.
  template <int dim>
  inline
  SymmetricTensor<2,dim>
  get_strain (const FEValues<dim> &fe_values,
              const unsigned int   shape_func,
              const unsigned int   q_point)
  {
    // Declare a temporary that will hold the return value:
    SymmetricTensor<2,dim> tmp;

    // First, fill diagonal terms which are simply the derivatives in
    // direction <code>i</code> of the <code>i</code> component of the
    // vector-valued shape function:
    for (unsigned int i=0; i<dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component (shape_func,q_point,i)[i];

    // Then fill the rest of the strain tensor. Note that since the tensor is
    // symmetric, we only have to compute one half (here: the upper right
    // corner) of the off-diagonal elements, and the implementation of the
    // <code>SymmetricTensor</code> class makes sure that at least to the
    // outside the symmetric entries are also filled (in practice, the class
    // of course stores only one copy). Here, we have picked the upper right
    // half of the tensor, but the lower left one would have been just as
    // good:
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
  

  // Finally, below we will need a function that computes the rotation matrix
  // induced by a displacement at a given point. In fact, of course, the
  // displacement at a single point only has a direction and a magnitude, it
  // is the change in direction and magnitude that induces rotations. In
  // effect, the rotation matrix can be computed from the gradients of a
  // displacement, or, more specifically, from the curl.
  //
  // The formulas by which the rotation matrices are determined are a little
  // awkward, especially in 3d. For 2d, there is a simpler way, so we
  // implement this function twice, once for 2d and once for 3d, so that we
  // can compile and use the program in both space dimensions if so desired --
  // after all, deal.II is all about dimension independent programming and
  // reuse of algorithm thoroughly tested with cheap computations in 2d, for
  // the more expensive computations in 3d. Here is one case, where we have to
  // implement different algorithms for 2d and 3d, but then can write the rest
  // of the program in a way that is independent of the space dimension.
  //
  // So, without further ado to the 2d implementation:
  Tensor<2,2>
  get_rotation_matrix (const std::vector<Tensor<1,2> > &grad_u)
  {
    // First, compute the curl of the velocity field from the gradients. Note
    // that we are in 2d, so the rotation is a scalar:
    const double curl = (grad_u[1][0] - grad_u[0][1]);

    // From this, compute the angle of rotation:
    const double angle = std::atan (curl);

    // And from this, build the antisymmetric rotation matrix:
    const double t[2][2] = {{ cos(angle), sin(angle) },
      {-sin(angle), cos(angle) }
    };
    return Tensor<2,2>(t);
  }


  // The 3d case is a little more contrived:
  Tensor<2,3>
  get_rotation_matrix (const std::vector<Tensor<1,3> > &grad_u)
  {
    // Again first compute the curl of the velocity field. This time, it is a
    // real vector:
    const Point<3> curl (grad_u[2][1] - grad_u[1][2],
                         grad_u[0][2] - grad_u[2][0],
                         grad_u[1][0] - grad_u[0][1]);

    // From this vector, using its magnitude, compute the tangent of the angle
    // of rotation, and from it the actual angle:
    const double tan_angle = std::sqrt(curl*curl);
    const double angle = std::atan (tan_angle);

    // Now, here's one problem: if the angle of rotation is too small, that
    // means that there is no rotation going on (for example a translational
    // motion). In that case, the rotation matrix is the identity matrix.
    //
    // The reason why we stress that is that in this case we have that
    // <code>tan_angle==0</code>. Further down, we need to divide by that
    // number in the computation of the axis of rotation, and we would get
    // into trouble when dividing doing so. Therefore, let's shortcut this and
    // simply return the identity matrix if the angle of rotation is really
    // small:
    if (angle < 1e-9)
      {
        static const double rotation[3][3]
        = {{ 1, 0, 0}, { 0, 1, 0 }, { 0, 0, 1 } };
        static const Tensor<2,3> rot(rotation);
        return rot;
      }

    // Otherwise compute the real rotation matrix. The algorithm for this is
    // not exactly obvious, but can be found in a number of books,
    // particularly on computer games where rotation is a very frequent
    // operation. Online, you can find a description at
    // http://www.makegames.com/3drotation/ and (this particular form, with
    // the signs as here) at
    // http://www.gamedev.net/reference/articles/article1199.asp:
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



  // @sect3{The <code>ElasticProblem</code> class template}

  // The main class is, except for its name, almost unchanged with respect to
  // the step-6 example.
  //
  // The only change is the use of a different class for the <code>fe</code>
  // variable: Instead of a concrete finite element class such as
  // <code>FE_Q</code>, we now use a more generic one,
  // <code>FESystem</code>. In fact, <code>FESystem</code> is not really a
  // finite element itself in that it does not implement shape functions of
  // its own.  Rather, it is a class that can be used to stack several other
  // elements together to form one vector-valued finite element. In our case,
  // we will compose the vector-valued element of <code>FE_Q(1)</code>
  // objects, as shown below in the constructor of this class.
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
    // At the end of each time step, we want to move the mesh vertices around
    // according to the incremental displacement computed in this time
    // step. This is the function in which this is done:
    void move_mesh ();
    
    // Next are two functions that handle the history variables stored in each
    // quadrature point. The first one is called before the first timestep to
    // set up a pristine state for the history variables. It only works on
    // those quadrature points on cells that belong to the present processor:
    void setup_quadrature_point_history ();

    // The second one updates the history variables at the end of each
    // timestep:
    void update_quadrature_point_history (const Vector<double>& displacement_update);

    void output_results () const;

    double compute_residual () const;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    FESystem<dim>        fe;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       system_rhs;

    Vector<double>       newton_update;
    Vector<double>       incremental_displacement;
    Vector<double>       solution;

    Vector<float>        estimated_error_per_cell;

    // The next block of variables is then related to the time dependent
    // nature of the problem: they denote the length of the time interval
    // which we want to simulate, the present time and number of time step,
    // and length of present timestep:
    double              present_time;
    double              present_timestep;
    double              end_time;
    unsigned int        timestep_no;

    // One difference of this program is that we declare the quadrature
    // formula in the class declaration. The reason is that in all the other
    // programs, it didn't do much harm if we had used different quadrature
    // formulas when computing the matrix and the right hand side, for
    // example. However, in the present case it does: we store information in
    // the quadrature points, so we have to make sure all parts of the program
    // agree on where they are and how many there are on each cell. Thus, let
    // us first declare the quadrature formula that will be used throughout...
    const QGauss<dim>   quadrature_formula;

    // ... and then also have a vector of history objects, one per quadrature
    // point on those cells for which we are responsible (i.e. we don't store
    // history data for quadrature points on cells that are owned by other
    // processors).
    std::vector<PointHistory<dim> > quadrature_point_history;

    // Finally, we have a static variable that denotes the linear relationship
    // between the stress and strain. Since it is a constant object that does
    // not depend on any input (at least not in this program), we make it a
    // static variable and will initialize it in the same place where we
    // define the constructor of this class:
    static const SymmetricTensor<4,dim> stress_strain_tensor;
  };


  // @sect3{The <code>BodyForce</code> class}

  // Before we go on to the main functionality of this program, we have to
  // define what forces will act on the body whose deformation we want to
  // study. These may either be body forces or boundary forces. Body forces
  // are generally mediated by one of the four basic physical types of forces:
  // gravity, strong and weak interaction, and electromagnetism. Unless one
  // wants to consider subatomic objects (for which quasistatic deformation is
  // irrelevant and an inappropriate description anyway), only gravity and
  // electromagnetic forces need to be considered. Let us, for simplicity
  // assume that our body has a certain mass density, but is either
  // non-magnetic and not electrically conducting or that there are no
  // significant electromagnetic fields around. In that case, the body forces
  // are simply <code>rho g</code>, where <code>rho</code> is the material
  // density and <code>g</code> is a vector in negative z-direction with
  // magnitude 9.81 m/s^2.  Both the density and <code>g</code> are defined in
  // the function, and we take as the density 7700 kg/m^3, a value commonly
  // assumed for steel.
  //
  // To be a little more general and to be able to do computations in 2d as
  // well, we realize that the body force is always a function returning a
  // <code>dim</code> dimensional vector. We assume that gravity acts along
  // the negative direction of the last, i.e. <code>dim-1</code>th
  // coordinate. The rest of the implementation of this function should be
  // mostly self-explanatory given similar definitions in previous example
  // programs. Note that the body force is independent of the location; to
  // avoid compiler warnings about unused function arguments, we therefore
  // comment out the name of the first argument of the
  // <code>vector_value</code> function:
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



  // @sect3{The <code>IncrementalBoundaryValue</code> class}

  // In addition to body forces, movement can be induced by boundary forces
  // and forced boundary displacement. The latter case is equivalent to forces
  // being chosen in such a way that they induce certain displacement.
  //
  // For quasistatic displacement, typical boundary forces would be pressure
  // on a body, or tangential friction against another body. We chose a
  // somewhat simpler case here: we prescribe a certain movement of (parts of)
  // the boundary, or at least of certain components of the displacement
  // vector. We describe this by another vector-valued function that, for a
  // given point on the boundary, returns the prescribed displacement.
  //
  // Since we have a time-dependent problem, the displacement increment of the
  // boundary equals the displacement accumulated during the length of the
  // timestep. The class therefore has to know both the present time and the
  // length of the present time step, and can then approximate the incremental
  // displacement as the present velocity times the present timestep.
  //
  // For the purposes of this program, we choose a simple form of boundary
  // displacement: we displace the top boundary with constant velocity
  // downwards. The rest of the boundary is either going to be fixed (and is
  // then described using an object of type <code>ZeroFunction</code>) or free
  // (Neumann-type, in which case nothing special has to be done).  The
  // implementation of the class describing the constant downward motion
  // should then be obvious using the knowledge we gained through all the
  // previous example programs:
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

    values = 0;
    values(dim-1) = -present_timestep * velocity;
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



  // @sect3{The <code>ElasticProblem</code> class implementation}

  // Now for the implementation of the main class. First, we initialize the
  // stress-strain tensor, which we have declared as a static const
  // variable. We chose Lame constants that are appropriate for steel:
  template <int dim>
  const SymmetricTensor<4,dim>
  ElasticProblem<dim>::stress_strain_tensor
    = get_stress_strain_tensor<dim> (/*lambda = */ 9.695e10,
                                                   /*mu     = */ 7.617e10);

  // @sect4{ElasticProblem::ElasticProblem}

  // Following is the constructor of the main class. As said before, we would
  // like to construct a vector-valued finite element that is composed of
  // several scalar finite elements (i.e., we want to build the vector-valued
  // element so that each of its vector components consists of the shape
  // functions of a scalar element). Of course, the number of scalar finite
  // elements we would like to stack together equals the number of components
  // the solution function has, which is <code>dim</code> since we consider
  // displacement in each space direction. The <code>FESystem</code> class can
  // handle this: we pass it the finite element of which we would like to
  // compose the system of, and how often it shall be repeated:

  template <int dim>
  ElasticProblem<dim>::ElasticProblem ()
    :
    dof_handler (triangulation),
    fe (FE_Q<dim>(1), dim),
    quadrature_formula (2)
  {}
  // In fact, the <code>FESystem</code> class has several more constructors
  // which can perform more complex operations than just stacking together
  // several scalar finite elements of the same type into one; we will get to
  // know these possibilities in later examples.


  // @sect4{ElasticProblem::~ElasticProblem}

  // The destructor, on the other hand, is exactly as in step-6:
  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }
  
  

  // @sect4{TopLevel::setup_quadrature_point_history}

  // At the beginning of our computations, we needed to set up initial values
  // of the history variables, such as the existing stresses in the material,
  // that we store in each quadrature point. As mentioned above, we use the
  // <code>user_pointer</code> for this that is available in each cell.
  //
  // To put this into larger perspective, we note that if we had previously
  // available stresses in our model (which we assume do not exist for the
  // purpose of this program), then we would need to interpolate the field of
  // preexisting stresses to the quadrature points. Likewise, if we were to
  // simulate elasto-plastic materials with hardening/softening, then we would
  // have to store additional history variables like the present yield stress
  // of the accumulated plastic strains in each quadrature
  // points. Pre-existing hardening or weakening would then be implemented by
  // interpolating these variables in the present function as well.
  template <int dim>
  void ElasticProblem<dim>::setup_quadrature_point_history ()
  {
    // Next, allocate as many quadrature objects as we need. Since the
    // <code>resize</code> function does not actually shrink the amount of
    // allocated memory if the requested new size is smaller than the old
    // size, we resort to a trick to first free all memory, and then
    // reallocate it: we declare an empty vector as a temporary variable and
    // then swap the contents of the old vector and this temporary
    // variable. This makes sure that the
    // <code>quadrature_point_history</code> is now really empty, and we can
    // let the temporary variable that now holds the previous contents of the
    // vector go out of scope and be destroyed. In the next step we can then
    // re-allocate as many elements as we need, with the vector
    // default-initializing the <code>PointHistory</code> objects, which
    // includes setting the stress variables to zero.
    {
      std::vector<PointHistory<dim> > tmp;
      tmp.swap (quadrature_point_history);
    }
    quadrature_point_history.resize (triangulation.n_active_cells() *
                                     quadrature_formula.size());

    // Finally loop over all cells again and set the user pointers from the
    // cells to point to the first quadrature point objects corresponding 
    // to this cell in the vector of such objects:
    unsigned int history_index = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
          {
            cell->set_user_pointer (&quadrature_point_history[history_index]);
            history_index += quadrature_formula.size();
          }

    // At the end, for good measure make sure that our count of elements was
    // correct and that we have both used up all objects we allocated
    // previously, and not point to any objects beyond the end of the
    // vector. Such defensive programming strategies are always good checks to
    // avoid accidental errors and to guard against future changes to this
    // function that forget to update all uses of a variable at the same
    // time. Recall that constructs using the <code>Assert</code> macro are
    // optimized away in optimized mode, so do not affect the run time of
    // optimized runs:
    Assert (history_index == quadrature_point_history.size(),
            ExcInternalError());
  }    
  
  
  // @sect4{MinimalSurfaceProblem::set_boundary_values}

  // The next function ensures that the solution vector's entries respect the
  // boundary values for our problem.  Having refined the mesh (or just
  // started computations), there might be new nodal points on the
  // boundary. These have values that are simply interpolated from the
  // previous mesh (or are just zero), instead of the correct boundary
  // values. This is fixed up by setting all boundary nodes explicit to the
  // right value:
  template <int dim>
  void ElasticProblem<dim>::set_boundary_values ()
  {
    // The way to describe this is as follows: for boundary indicator zero
    // (bottom face) we use a dim-dimensional zero function representing no
    // motion in any coordinate direction. For the boundary with indicator 1
    // (top surface), we use the <code>IncrementalBoundaryValues</code> class,
    // but we specify an additional argument to the
    // <code>VectorTools::interpolate_boundary_values</code> function denoting
    // which vector components it should apply to; this is a vector of bools
    // for each vector component and because we only want to restrict vertical
    // motion, it has only its last component set:
    FEValuesExtractors::Scalar h_component (dim-2);
    FEValuesExtractors::Scalar v_component (dim-1);
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 1,
                                 ZeroFunction<dim> (dim),
                                 boundary_values);
//    VectorTools::
//    interpolate_boundary_values (dof_handler,
//                                 2,
//                                 ZeroFunction<dim> (dim),
//                                 boundary_values,
//                                 fe.component_mask(h_component));
    VectorTools::
    interpolate_boundary_values (dof_handler,
                                 2,
                                 IncrementalBoundaryValues<dim>(present_time,
                                                                present_timestep),
                                 boundary_values,
                                 fe.component_mask(v_component));
                                        
    for (std::map<types::global_dof_index, double>::const_iterator
       p = boundary_values.begin();
       p != boundary_values.end(); ++p)
         incremental_displacement(p->first) = p->second;
  }


  // @sect4{TopLevel::assemble_system}

  // Again, assembling the system matrix and right hand side follows the same
  // structure as in many example programs before. In particular, it is mostly
  // equivalent to step-17, except for the different right hand side that now
  // only has to take into account internal stresses. In addition, assembling
  // the matrix is made significantly more transparent by using the
  // <code>SymmetricTensor</code> class: note the elegance of forming the
  // scalar products of symmetric tensors of rank 2 and 4. The implementation
  // is also more general since it is independent of the fact that we may or
  // may not be using an isotropic elasticity tensor.
  //
  // The first part of the assembly routine is as always:
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
                                                    
//    std::cout << std::endl 
//          << "beg - norm of rhs is " << system_rhs.l2_norm()
//          << std::endl;                                                    

    // As in step-17, we only need to loop over all cells that belong to the
    // present processor:
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
        {
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);

          // In the case of a Newton method, the operator assembled here
          // is the directional derivative of the residue in the direction
          // of the iteration displacement increment 'newton_update'. It
          // can be dependent on the tangent or the secant stiffness tensor.
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              for (unsigned int q_point=0; q_point<n_q_points;
                   ++q_point)
                {
                  const SymmetricTensor<2,dim>
                  eps_phi_i = get_strain (fe_values, i, q_point),
                  eps_phi_j = get_strain (fe_values, j, q_point);

                  // Make use of previous iteration tangent/secant 
                  // stress-strain tensor
                  cell_matrix(i,j)
                  += (eps_phi_i * stress_strain_tensor * eps_phi_j
                      *
                      fe_values.JxW (q_point));
                }


          // Then also assemble the local right hand side contributions. For
          // this, we need to access the prior stress value in this quadrature
          // point. To get it, we use the user pointer of this cell that
          // points into the global array to the quadrature point data
          // corresponding to the first quadrature point of the present cell,
          // and then add an offset corresponding to the index of the
          // quadrature point we presently consider:
          const PointHistory<dim> *local_quadrature_points_data
            = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
          // In addition, we need the values of the external body forces at
          // the quadrature points on this cell:
          body_force.vector_value_list (fe_values.get_quadrature_points(),
                                        body_force_values);
          // Then we can loop over all degrees of freedom on this cell and
          // compute local contributions to the right hand side:
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;

              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  const SymmetricTensor<2,dim> &old_stress
                    = local_quadrature_points_data[q_point].old_stress;

                  cell_rhs(i) += (body_force_values[q_point](component_i) *
                                  fe_values.shape_value (i,q_point)
                                  -
                                  old_stress *
                                  get_strain (fe_values,i,q_point))
                                 *
                                 fe_values.JxW (q_point);
                }
            }

          // Now that we have the local contributions to the linear system, we
          // need to transfer it into the global objects. This is done exactly
          // as in step-17:
          cell->get_dof_indices (local_dof_indices);

          hanging_node_constraints
          .distribute_local_to_global (cell_matrix, cell_rhs,
                                       local_dof_indices,
                                       system_matrix, system_rhs);
        }
    
//    std::cout << "mid - norm of rhs is " << system_rhs.l2_norm()
//          << std::endl;    

    // Now compress the vector and the system matrix:
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    // Finally, we remove hanging nodes from the system and apply zero
    // boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    FEValuesExtractors::Scalar h_component (dim-2);
    FEValuesExtractors::Scalar v_component (dim-1);
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              ZeroFunction<dim>(dim),
                                              boundary_values);
//    VectorTools::interpolate_boundary_values (dof_handler,
//                                              2,
//                                              ZeroFunction<dim>(dim),
//                                              boundary_values,
//                                              fe.component_mask(h_component)); 
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
                                              ZeroFunction<dim>(dim),
                                              boundary_values,
                                              fe.component_mask(v_component)); 
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        newton_update,
                                        system_rhs);
                                        
//    std::cout << "end - norm of rhs is " << system_rhs.l2_norm()
//          << std::endl;  
  }


  // @sect4{TopLevel::solve_linear_problem}

  template <int dim>
  unsigned int ElasticProblem<dim>::solve_linear_problem ()
  {
    SolverControl       solver_control (1000,
                                            1e-16*system_rhs.l2_norm());
    SolverCG<>          cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
         
    cg.solve (system_matrix, newton_update, system_rhs,
                  preconditioner);

    hanging_node_constraints.distribute (newton_update);

    const double alpha = determine_step_length();
    incremental_displacement.add (alpha, newton_update);

    return solver_control.last_step();
  }


  // @sect4{TopLevel::update_quadrature_point_history}

  // At the end of each time step, we should have computed an incremental
  // displacement update so that the material in its new configuration
  // accommodates for the difference between the external body and boundary
  // forces applied during this time step minus the forces exerted through
  // preexisting internal stresses. In order to have the preexisting
  // stresses available at the next time step, we therefore have to update the
  // preexisting stresses with the stresses due to the incremental
  // displacement computed during the present time step. Ideally, the
  // resulting sum of internal stresses would exactly counter all external
  // forces. Indeed, a simple experiment can make sure that this is so: if we
  // choose boundary conditions and body forces to be time independent, then
  // the forcing terms (the sum of external forces and internal stresses)
  // should be exactly zero. If you make this experiment, you will realize
  // from the output of the norm of the right hand side in each time step that
  // this is almost the case: it is not exactly zero, since in the first time
  // step the incremental displacement and stress updates were computed
  // relative to the undeformed mesh, which was then deformed. In the second
  // time step, we again compute displacement and stress updates, but this
  // time in the deformed mesh -- there, the resulting updates are very small
  // but not quite zero. This can be iterated, and in each such iteration the
  // residual, i.e. the norm of the right hand side vector, is reduced; if one
  // makes this little experiment, one realizes that the norm of this residual
  // decays exponentially with the number of iterations, and after an initial
  // very rapid decline is reduced by roughly a factor of about 3.5 in each
  // iteration (for one testcase I looked at, other testcases, and other
  // numbers of unknowns change the factor, but not the exponential decay).

  // In a sense, this can then be considered as a quasi-timestepping scheme to
  // resolve the nonlinear problem of solving large-deformation elasticity on
  // a mesh that is moved along in a Lagrangian manner.
  template <int dim>
  void ElasticProblem<dim>::update_quadrature_point_history 
        (const Vector<double>& displacement_update)
  {
    // First, set up an <code>FEValues</code> object by which we will evaluate
    // the displacements and the gradients thereof at the
    // quadrature points, together with a vector that will hold this
    // information:
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values | update_gradients);
    std::vector<std::vector<Tensor<1,dim> > >
    displacement_update_grads (quadrature_formula.size(),
                                  std::vector<Tensor<1,dim> >(dim));

    // Verify that the previously initialized quadrature points history vector
    // has been correctly loaded:
    Assert (quadrature_point_history.size() > 0,
            ExcInternalError());

    // Then loop over all cells and do the job in the cells that belong to our
    // subdomain:
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      //if (cell->is_locally_owned())
          {
          PointHistory<dim> *local_quadrature_points_history
            = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
          Assert (local_quadrature_points_history >=
                  &quadrature_point_history.front(),
                  ExcInternalError());
          Assert (local_quadrature_points_history <
                  &quadrature_point_history.back(),
                  ExcInternalError());
          // Then initialize the <code>FEValues</code> object on the present
          // cell, and extract the gradients of the displacement at the
          // quadrature points for later computation of the strains
          fe_values.reinit (cell);
          fe_values.get_function_gradients (displacement_update,
                                            displacement_update_grads);

          // Then loop over the quadrature points of this cell:
          for (unsigned int q=0; q<quadrature_formula.size(); ++q)
            {
              // On each quadrature point, compute the strain from
              // the gradients, and multiply it by the stress-strain tensor to
              // get the stress update. Then add this update to the already
              // existing strain at this point:
              const SymmetricTensor<2,dim> new_stress
                = (local_quadrature_points_history[q].old_stress
                   +
                   (stress_strain_tensor *
                    get_strain (displacement_update_grads[q])));
                    
              // Finally, we have to rotate the result. For this, we first
              // have to compute a rotation matrix at the present quadrature
              // point from the incremental displacements. In fact, it can be
              // computed from the gradients, and we already have a function
              // for that purpose:
              const Tensor<2,dim> rotation
                = get_rotation_matrix (displacement_update_grads[q]);
              // Note that the result, a rotation matrix, is in general an
              // antisymmetric tensor of rank 2, so we must store it as a full
              // tensor.

              // With this rotation matrix, we can compute the rotated tensor
              // by contraction from the left and right, after we expand the
              // symmetric tensor <code>new_stress</code> into a full tensor:
              const SymmetricTensor<2,dim> rotated_new_stress
                = symmetrize(transpose(rotation) *
                             static_cast<Tensor<2,dim> >(new_stress) *
                             rotation);
              // Note that while the result of the multiplication of these
              // three matrices should be symmetric, it is not due to floating
              // point round off: we get an asymmetry on the order of 1e-16 of
              // the off-diagonal elements of the result. When assigning the
              // result to a <code>SymmetricTensor</code>, the constructor of
              // that class checks the symmetry and realizes that it isn't
              // exactly symmetric; it will then raise an exception. To avoid
              // that, we explicitly symmetrize the result to make it exactly
              // symmetric.

              // The result of all these operations is then written back into
              // the original place:
              local_quadrature_points_history[q].old_stress
                = rotated_new_stress;
              local_quadrature_points_history[q].old_norm = rotated_new_stress.norm();
            }
          }
  }


  // @sect4{ElasticProblem::compute_residual}
  // In order to monitor convergence, we need a way to compute the norm of the
  // (discrete) residual, i.e., the norm of the vector
  // $\left<F(u^n),\varphi_i\right>$ with $F(u)=-\nabla \cdot \left(
  // \frac{1}{\sqrt{1+|\nabla u|^{2}}}\nabla u \right)$ as discussed in the
  // introduction. It turns out that (although we don't use this feature in
  // the current version of the program) one needs to compute the residual
  // $\left<F(u^n+\alpha^n\;\delta u^n),\varphi_i\right>$ when determining
  // optimal step lengths, and so this is what we implement here: the function
  // takes the step length $\alpha^n$ as an argument. The original
  // functionality is of course obtained by passing a zero as argument.
  //
  // In the function below, we first set up a vector for the residual, and
  // then a vector for the evaluation point $u^n+\alpha^n\;\delta u^n$. This
  // is followed by the same boilerplate code we use for all integration
  // operations:
  template <int dim>
  double ElasticProblem<dim>::compute_residual () const
  {
    Vector<double> residual (dof_handler.n_dofs());
    
    residual = 0;
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    BodyForce<dim>      body_force;
    std::vector<Vector<double> > body_force_values (n_q_points,
                                                    Vector<double>(dim));

    Vector<double>               cell_residual (dofs_per_cell);

//    std::cout << "beg - norm of res is " << residual.l2_norm()
//          << std::endl;  

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
        {
          cell_residual = 0;
          fe_values.reinit (cell);

          // Then also assemble the local right hand side contributions. For
          // this, we need to access the prior stress value in this quadrature
          // point. To get it, we use the user pointer of this cell that
          // points into the global array to the quadrature point data
          // corresponding to the first quadrature point of the present cell,
          // and then add an offset corresponding to the index of the
          // quadrature point we presently consider:
          const PointHistory<dim> *local_quadrature_points_data
            = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
          // In addition, we need the values of the external body forces at
          // the quadrature points on this cell:
          body_force.vector_value_list (fe_values.get_quadrature_points(),
                                        body_force_values);
          // Then we can loop over all degrees of freedom on this cell and
          // compute local contributions to the right hand side:
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;

              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  const SymmetricTensor<2,dim> &old_stress
                    = local_quadrature_points_data[q_point].old_stress;

                  cell_residual(i) += (body_force_values[q_point](component_i) *
                                       fe_values.shape_value (i,q_point)
                                       -
                                       old_stress *
                                       get_strain (fe_values,i,q_point))
                                      *
                                      fe_values.JxW (q_point);
                }
            }

          // Now that we have the local contributions to the linear system, we
          // need to transfer it into the global objects. This is done exactly
          // as in step-17:
          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i) 
                residual(local_dof_indices[i]) += cell_residual(i);
        }

//    std::cout << "mid - norm of res is " << residual.l2_norm()
//          << std::endl;  

    // At the end of this function we also have to deal with the hanging node
    // constraints and with the issue of boundary values. With regard to the
    // latter, we have to set to zero the elements of the residual vector for
    // all entries that correspond to degrees of freedom that sit at the
    // boundary. The reason is that because the value of the solution there is
    // fixed, they are of course no "real" degrees of freedom and so, strictly
    // speaking, we shouldn't have assembled entries in the residual vector
    // for them. However, as we always do, we want to do exactly the same
    // thing on every cell and so we didn't not want to deal with the question
    // of whether a particular degree of freedom sits at the boundary in the
    // integration above. Rather, we will simply set to zero these entries
    // after the fact. To this end, we first need to determine which degrees
    // of freedom do in fact belong to the boundary and then loop over all of
    // those and set the residual entry to zero. This happens in the following
    // lines which we have already seen used in step-11:
    hanging_node_constraints.condense (residual);

    std::vector<bool> boundary_dofs (dof_handler.n_dofs());
    DoFTools::extract_boundary_dofs (dof_handler,
                                     ComponentMask(),
                                     boundary_dofs);
    for (unsigned int i=0; i<dof_handler.n_dofs(); ++i)
      if (boundary_dofs[i] == true)
        residual(i) = 0;

//    std::cout << "end - norm of res is " << residual.l2_norm()
//          << std::endl;  

    // At the end of the function, we return the norm of the residual:
    return residual.l2_norm();
  }


  // @sect4{TopLevel::solve_timestep}

  // The next function is the one that controls what all has to happen within
  // a timestep. The order of things should be relatively self-explanatory
  // from the function names:
  template <int dim>
  void ElasticProblem<dim>::solve_timestep ()
  {    
    double previous_res;
    
    // The Newton iteration starts next. During the first step we do not have
    // information about the residual prior to this step and so we continue
    // the Newton iteration until we have reached at least one iteration and
    // until residual is less than $10^{-3}$.
    //
    // At the beginning of the loop, we do a bit of setup work. In the first
    // go around, we compute the solution on the twice globally refined mesh
    // after setting up the basic data structures and ensuring that the first
    // Newton iterate already has the correct boundary values. In all
    // following mesh refinement loops, the mesh will be refined adaptively.    
    do
      { 
        // On every mesh we do exactly five Newton steps. We print the initial
        // residual here and then start the iterations on this mesh.
        //
        // In every Newton step the system matrix and the right hand side have
        // to be computed first, after which we store the norm of the right
        // hand side as the residual to check against when deciding whether to
        // stop the iterations. We then solve the linear system (the function
        // also updates $u^{n+1}=u^n+\alpha^n\;\delta u^n$) and output the
        // residual at the end of this Newton step:
        std::cout << "  Initial residual: "
                  << compute_residual()
                  << std::endl;

        for (unsigned int inner_iteration=0; inner_iteration<5; ++inner_iteration)
          {

            std::cout << "    Assembling system..." << std::flush;
            assemble_system ();
            previous_res = system_rhs.l2_norm();
            
            compute_residual();

            const unsigned int n_iterations = solve_linear_problem ();
            
            std::cout << "    Solver - norm of rhs is " << system_rhs.l2_norm()
                  << std::endl;  
            std::cout << "    Solver - norm of newton update is " << newton_update.l2_norm()
                  << std::endl;  
            std::cout << "    Solver converged in " << n_iterations
                  << " iterations." << std::endl;
            
            std::cout << "    Updating quadrature point data..." << std::flush;
            update_quadrature_point_history (newton_update);
            std::cout << std::endl;
            
            previous_res = compute_residual();

            std::cout << "  Residual: "
                      << previous_res
                      << std::endl
                      << "  -"
                      << std::endl;
          }
      } while (previous_res>1e-3);
  }



  // @sect4{ElasticProblem::error_estimation}

  // The quadrature formula is adapted to the linear elements
  // again. Note that the error estimator by default adds up the estimated
  // obtained from all components of the finite element solution, i.e., it
  // uses the displacement in all directions with the same weight.
  template <int dim>
  void ElasticProblem<dim>::error_estimation ()
  {
    estimated_error_per_cell.reinit(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(2),
                                        typename FunctionMap<dim>::type(),
                                        incremental_displacement,
                                        estimated_error_per_cell);
  }



  // @sect4{MinimalSurfaceProblem::determine_step_length}

  // As discussed in the introduction, Newton's method frequently does not
  // converge if we always take full steps, i.e., compute $u^{n+1}=u^n+\delta
  // u^n$. Rather, one needs a damping parameter (step length) $\alpha^n$ and
  // set $u^{n+1}=u^n+\alpha^n\delta u^n$. This function is the one called
  // to compute $\alpha^n$.
  //
  // Here, we simply always return 0.1. This is of course a sub-optimal
  // choice: ideally, what one wants is that the step size goes to one as we
  // get closer to the solution, so that we get to enjoy the rapid quadratic
  // convergence of Newton's method. We will discuss better strategies below
  // in the results section.
  template <int dim>
  double ElasticProblem<dim>::determine_step_length() const
  {
    return 1.0;
  }
  
  
  
  // @sect4{TopLevel::move_mesh}

  // At the end of each time step, we move the nodes of the mesh according to
  // the incremental displacements computed in this time step. To do this, we
  // keep a vector of flags that indicate for each vertex whether we have
  // already moved it around, and then loop over all cells and move those
  // vertices of the cell that have not been moved yet. It is worth noting
  // that it does not matter from which of the cells adjacent to a vertex we
  // move this vertex: since we compute the displacement using a continuous
  // finite element, the displacement field is continuous as well and we can
  // compute the displacement of a given vertex from each of the adjacent
  // cells. We only have to make sure that we move each node exactly once,
  // which is why we keep the vector of flags.
  //
  // There are two noteworthy things in this function. First, how we get the
  // displacement field at a given vertex using the
  // <code>cell-@>vertex_dof_index(v,d)</code> function that returns the index
  // of the <code>d</code>th degree of freedom at vertex <code>v</code> of the
  // given cell. In the present case, displacement in the k-th coordinate
  // direction corresponds to the k-th component of the finite element. Using a
  // function like this bears a certain risk, because it uses knowledge of the
  // order of elements that we have taken together for this program in the
  // <code>FESystem</code> element. If we decided to add an additional
  // variable, for example a pressure variable for stabilization, and happened
  // to insert it as the first variable of the element, then the computation
  // below will start to produce nonsensical results. In addition, this
  // computation rests on other assumptions: first, that the element we use
  // has, indeed, degrees of freedom that are associated with vertices. This
  // is indeed the case for the present Q1 element, as would be for all Qp
  // elements of polynomial order <code>p</code>. However, it would not hold
  // for discontinuous elements, or elements for mixed formulations. Secondly,
  // it also rests on the assumption that the displacement at a vertex is
  // determined solely by the value of the degree of freedom associated with
  // this vertex; in other words, all shape functions corresponding to other
  // degrees of freedom are zero at this particular vertex. Again, this is the
  // case for the present element, but is not so for all elements that are
  // presently available in deal.II. Despite its risks, we choose to use this
  // way in order to present a way to query individual degrees of freedom
  // associated with vertices.
  //
  // In this context, it is instructive to point out what a more general way
  // would be. For general finite elements, the way to go would be to take a
  // quadrature formula with the quadrature points in the vertices of a
  // cell. The <code>QTrapez</code> formula for the trapezoidal rule does
  // exactly this. With this quadrature formula, we would then initialize an
  // <code>FEValues</code> object in each cell, and use the
  // <code>FEValues::get_function_values</code> function to obtain the values
  // of the solution function in the quadrature points, i.e. the vertices of
  // the cell. These are the only values that we really need, i.e. we are not
  // at all interested in the weights (or the <code>JxW</code> values)
  // associated with this particular quadrature formula, and this can be
  // specified as the last argument in the constructor to
  // <code>FEValues</code>. The only point of minor inconvenience in this
  // scheme is that we have to figure out which quadrature point corresponds
  // to the vertex we consider at present, as they may or may not be ordered
  // in the same order.
  //
  // This inconvenience could be avoided if finite elements have support
  // points on vertices (which the one here has; for the concept of support
  // points, see @ref GlossSupport "support points"). For such a case, one
  // could construct a custom quadrature rule using
  // FiniteElement::get_unit_support_points(). The first
  // <code>GeometryInfo@<dim@>::vertices_per_cell*fe.dofs_per_vertex</code>
  // quadrature points will then correspond to the vertices of the cell and
  // are ordered consistent with <code>cell-@>vertex(i)</code>, taking into
  // account that support points for vector elements will be duplicated
  // <code>fe.dofs_per_vertex</code> times.
  //
  // Another point worth explaining about this short function is the way in
  // which the triangulation class exports information about its vertices:
  // through the <code>Triangulation::n_vertices</code> function, it
  // advertises how many vertices there are in the triangulation. Not all of
  // them are actually in use all the time -- some are left-overs from cells
  // that have been coarsened previously and remain in existence since deal.II
  // never changes the number of a vertex once it has come into existence,
  // even if vertices with lower number go away. Secondly, the location
  // returned by <code>cell-@>vertex(v)</code> is not only a read-only object
  // of type <code>Point@<dim@></code>, but in fact a reference that can be
  // written to. This allows to move around the nodes of a mesh with relative
  // ease, but it is worth pointing out that it is the responsibility of an
  // application program using this feature to make sure that the resulting
  // cells are still useful, i.e. are not distorted so much that the cell is
  // degenerated (indicated, for example, by negative Jacobians). Note that we
  // do not have any provisions in this function to actually ensure this, we
  // just have faith.
  //
  // After this lengthy introduction, here are the full 20 or so lines of
  // code:
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
  
  

    // @sect4{TopLevel::output_results}

  // This function generates the graphical output in .vtu format as explained
  // in the introduction. Each process will only work on the cells it owns,
  // and then write the result into a file of its own. Additionally, processor
  // 0 will write the record files the reference all the .vtu files.
  //
  // The crucial part of this function is to give the <code>DataOut</code>
  // class a way to only work on the cells that the present process owns.

  template <int dim>
  void ElasticProblem<dim>::output_results () const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);

    std::vector<std::string> 
        solution_names (dim, "displacement");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
       data_component_interpretation
       (dim, DataComponentInterpretation::component_is_part_of_vector);
    // After setting up the names for the different components of the solution
    // vector, we can add the solution vector to the list of data vectors
    // scheduled for output. Note that the following function takes a vector
    // of strings as second argument, whereas the one which we have used in
    // all previous examples accepted a string there. In fact, the latter
    // function is only a shortcut for the function which we call here: it
    // puts the single string that is passed to it into a vector of strings
    // with only one element and forwards that to the other function.

    data_out.add_data_vector (solution, 
                              solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
                               
    data_out.add_data_vector (estimated_error_per_cell, "error_per_cell");
    
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
        {
          fe_values.reinit (cell);
          fe_values.get_function_gradients (solution,
                                            solution_grads);
          SymmetricTensor<2,dim> accumulated_strain;
          
          // Then loop over the quadrature points of this cell:
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
    }
    // Finally attach this vector as well to be treated for output:
    data_out.add_data_vector (norm_of_strain, "norm_of_strain");

    // The next thing is that we wanted to output something like the average
    // norm of the stresses that we have stored in each cell. This may seem
    // complicated, since on the present processor we only store the stresses
    // in quadrature points on those cells that actually belong to the present
    // process. In other words, it seems as if we can't compute the average
    // stresses for all cells. However, remember that our class derived from
    // <code>DataOut</code> only iterates over those cells that actually do
    // belong to the present processor, i.e. we don't have to compute anything
    // for all the other cells as this information would not be touched. The
    // following little loop does this. We enclose the entire block into a
    // pair of braces to make sure that the iterator variables do not remain
    // accidentally visible beyond the end of the block in which they are
    // used:
    Vector<double> norm_of_stress (triangulation.n_active_cells());
    {
      // Loop over all the cells...
      typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for (; cell!=endc; ++cell)
//        if (cell->is_locally_owned())
          {
            // On these cells, add up the stresses over all quadrature
            // points...
            SymmetricTensor<2,dim> accumulated_stress;
            for (unsigned int q=0;
                 q<quadrature_formula.size();
                 ++q)
              accumulated_stress +=
                reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q]
                .old_stress;

            // ...then write the norm of the average to their destination:
            norm_of_stress(cell->active_cell_index())
              = (accumulated_stress /
                 quadrature_formula.size()).norm();
          }
      // And on the cells that we are not interested in, set the respective
      // value in the vector to a bogus value (norms must be positive, and a
      // large negative value should catch your eye) in order to make sure
      // that if we were somehow wrong about our assumption that these
      // elements would not appear in the output file, that we would find out
      // by looking at the graphical output:
//        else
//          norm_of_stress(cell->active_cell_index()) = -1e+20;
    }
    // Finally attach this vector as well to be treated for output:
    data_out.add_data_vector (norm_of_stress, "norm_of_stress");

//    // As a last piece of data, let us also add the partitioning of the domain
//    // into subdomains associated with the processors if this is a parallel
//    // job. This works in the exact same way as in the step-17 program:
//    std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());
//    GridTools::get_subdomain_association (triangulation, partition_int);
//    const Vector<double> partitioning(partition_int.begin(),
//                                      partition_int.end());
//    data_out.add_data_vector (partitioning, "partitioning");

    // Finally, with all this data, we can instruct deal.II to munge the
    // information and produce some intermediate data structures that contain
    // all these solution and other data vectors:
    data_out.build_patches ();


    // Let us determine the name of the file we will want to write it to. We
    // compose it of the prefix <code>solution-</code>, followed by the time
    // step number, and finally the processor id (encoded as a three digit
    // number):
    std::string filename = "solution-" + Utilities::int_to_string(timestep_no,4)
//                           + "." + Utilities::int_to_string(this_mpi_process,3)
                           + ".vtu";

    // The following assertion makes sure that there are less than 1000
    // processes (a very conservative check, but worth having anyway) as our
    // scheme of generating process numbers would overflow if there were 1000
    // processes or more. Note that we choose to use <code>AssertThrow</code>
    // rather than <code>Assert</code> since the number of processes is a
    // variable that depends on input files or the way the process is started,
    // rather than static assumptions in the program code. Therefore, it is
    // inappropriate to use <code>Assert</code> that is optimized away in
    // optimized mode, whereas here we actually can assume that users will run
    // the largest computations with the most processors in optimized mode,
    // and we should check our assumptions in this particular case, and not
    // only when running in debug mode:
//    AssertThrow (n_mpi_processes < 1000, ExcNotImplemented());

    // With the so-completed filename, let us open a file and write the data
    // we have generated into it:
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

//    // The record files must be written only once and not by each processor,
//    // so we do this on processor 0:
//    if (this_mpi_process==0)
//      {
        // Here we collect all filenames of the current timestep (same format as above)
        std::vector<std::string> filenames;
//        for (unsigned int i=0; i<n_mpi_processes; ++i)
          filenames.push_back ("solution-" + Utilities::int_to_string(timestep_no,4)
//                               + "." + Utilities::int_to_string(i,3)
                               + ".vtu");

        // Now we write the .visit file. The naming is similar to the .vtu files, only
        // that the file obviously doesn't contain a processor id.
        const std::string
        visit_master_filename = ("solution-" +
                                 Utilities::int_to_string(timestep_no,4) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        data_out.write_visit_record (visit_master, filenames);

        // Similarly, we write the paraview .pvtu:
        const std::string
        pvtu_master_filename = ("solution-" +
                                Utilities::int_to_string(timestep_no,4) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);

        // Finally, we write the paraview record, that references all .pvtu files and
        // their respective time. Note that the variable times_and_names is declared
        // static, so it will retain the entries from the pervious timesteps.
        static std::vector<std::pair<double,std::string> > times_and_names;
        times_and_names.push_back (std::pair<double,std::string> (present_time, pvtu_master_filename));
        std::ofstream pvd_output ("solution.pvd");
        data_out.write_pvd_record (pvd_output, times_and_names);
//      }

  }


  // @sect4{ElasticProblem::make_grid}

  // The function that does the creation of the L-shaped grid and defines the 
  // ids of the different boundary regions for the later application of boundary
  // conditions.

  template <int dim>
  void ElasticProblem<dim>::make_grid ()
  {
    GridGenerator::hyper_L (triangulation, -1, 1);
    triangulation.refine_global (3);
    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
             cell != triangulation.end(); 
                ++cell)
       for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
           {
             if (cell->face(f)->center()[0] == 1.0)
                cell->face(f)->set_boundary_id (1);
             if (cell->face(f)->center()[1] == 1.0)
                cell->face(f)->set_boundary_id (2);
           }
  }


  // @sect4{ElasticProblem::setup_system}

  // Setting up the system of equations is identical to the function used in
  // the step-6 example. The <code>DoFHandler</code> class and all other
  // classes used here are fully aware that the finite element we want to use
  // is vector-valued, and take care of the vector-valuedness of the finite
  // element themselves. (In fact, they do not, but this does not need to
  // bother you: since they only need to know how many degrees of freedom
  // there are per vertex, line and cell, and they do not ask what they
  // represent, i.e. whether the finite element under consideration is
  // vector-valued or whether it is, for example, a scalar Hermite element
  // with several degrees of freedom on each vertex).
  template <int dim>
  void ElasticProblem<dim>::setup_system ()
  {
    dof_handler.distribute_dofs (fe);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);

    incremental_displacement.reinit (dof_handler.n_dofs());
    
    newton_update.reinit (dof_handler.n_dofs());
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    
    setup_quadrature_point_history ();
  }
  
  
  // @sect4{TopLevel::do_timestep}

  // Subsequent timesteps are simpler, and probably do not require any more
  // documentation given the explanations for the previous function above:
  template <int dim>
  void ElasticProblem<dim>::do_timestep ()
  {
//    double previous_res;
    
    present_time += present_timestep;
    ++timestep_no;
    std::cout << "Timestep " << timestep_no << " at time " << present_time
          << std::endl;
    if (present_time > end_time)
      {
        present_timestep -= (present_time - end_time);
        present_time = end_time;
      }

    incremental_displacement = 0;
    
    // Application of correct boundary displacement on
    // incremental_displacement
    set_boundary_values ();
    update_quadrature_point_history (incremental_displacement);
    
//    incremental_displacement = newton_update;
    
    solve_timestep ();

    solution+=incremental_displacement;

    error_estimation ();

//    move_mesh ();
    output_results ();

    std::cout << std::endl;
  }
 
  
  
  // The last of the public functions is the one that directs all the work,
  // <code>run()</code>. It initializes the variables that describe where in
  // time we presently are, then runs the first time step, then loops over all
  // the other time steps. Note that for simplicity we use a fixed time step,
  // whereas a more sophisticated program would of course have to choose it in
  // some more reasonable way adaptively:
  template <int dim>
  void ElasticProblem<dim>::run ()
  {
    present_time = 0;
    present_timestep = 1;
    end_time = 10;
    timestep_no = 0;
    
    make_grid ();

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;
              
    setup_system ();

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;                 

    while (present_time < end_time)
      do_timestep ();
  }
}  


// @sect3{The <code>main</code> function}

// After closing the <code>Step8</code> namespace in the last line above, the
// following is the main function of the program and is again exactly like in
// step-6 (apart from the changed class names, of course).
int main ()
{
  try
    {
      Step8::ElasticProblem<2> elastic_problem;
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
