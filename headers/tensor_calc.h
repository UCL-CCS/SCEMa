#ifndef TENSOR_CALC_H
#define TENSOR_CALC_H

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/symmetric_tensor.h>

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

using namespace dealii;

template <int dim>
inline
Tensor<2,dim>
compute_rotation_tensor (Tensor<1,dim> vorig, Tensor<1,dim> vdest)
{
	Tensor<2,dim> rotam;

	// Filling identity matrix
	Tensor<2,dim> idmat;
	idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;

	// Decalaration variables rotation matrix computation
	double ccos;
	Tensor<2,dim> skew_rot;

	// Compute the scalar product of the local and global vectors
	ccos = scalar_product(vorig, vdest);

	// Filling the skew-symmetric cross product matrix (a^Tb-b^Ta)
	for (unsigned int i=0; i<dim; ++i)
		for (unsigned int j=0; j<dim; ++j)
			skew_rot[i][j] = vorig[j]*vdest[i] - vorig[i]*vdest[j];

	// Assembling the rotation matrix
	rotam = idmat + skew_rot + (1/(1+ccos))*skew_rot*skew_rot;

	return rotam;
}

template <int dim>
inline
SymmetricTensor<2,dim>
rotate_tensor (const SymmetricTensor<2,dim> &tensor,
		const Tensor<2,dim> &rotam)
{
	SymmetricTensor<2,dim> stmp;

	Tensor<2,dim> tmp;

	Tensor<2,dim> tmp_tensor = tensor;

	tmp = rotam*tmp_tensor*transpose(rotam);

	for(unsigned int k=0;k<dim;k++)
		for(unsigned int l=k;l<dim;l++)
			stmp[k][l] = 0.5*(tmp[k][l] + tmp[l][k]);

	return stmp;
}

template <int dim>
inline
SymmetricTensor<4,dim>
rotate_tensor (const SymmetricTensor<4,dim> &tensor,
		const Tensor<2,dim> &rotam)
{
	SymmetricTensor<4,dim> tmp;
	tmp = 0;

	// Loop over the indices of the SymmetricTensor (upper "triangle" only)
	for(unsigned int k=0;k<dim;k++)
		for(unsigned int l=k;l<dim;l++)
			for(unsigned int s=0;s<dim;s++)
				for(unsigned int t=s;t<dim;t++)
				{
					for(unsigned int m=0;m<dim;m++)
						for(unsigned int n=0;n<dim;n++)
							for(unsigned int p=0;p<dim;p++)
								for(unsigned int r=0;r<dim;r++)
									tmp[k][l][s][t] +=
											tensor[m][n][p][r]
											* rotam[k][m] * rotam[l][n]
											* rotam[s][p] * rotam[t][r];
				}

	return tmp;
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

#endif
