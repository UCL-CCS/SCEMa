#ifndef MATHISTPREDICT_STRAIN2SPLINE_H
#define MATHISTPREDICT_STRAIN2SPLINE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>

#include "spline.h"

class Strain6D
{
	public:
		uint32_t num_points;
		std::vector<double> XX, YY, ZZ, XY, XZ, YZ;
		std::string in_fname;

		Strain6D()
		{
			num_points_read_in = 0;
			num_points = 0;
		}

		void from_file(const char *in_fname)
		{
			this->in_fname = std::string(in_fname);
			std::ifstream infile(in_fname);
			if(infile.fail()) {
				fprintf(stderr, "Could not open %s for reading.\n", in_fname);
				exit(1);
			}
			double xx, yy, zz, xy, xz, yz;
			while (infile >> xx >> yy >> zz >> xy >> xz >> yz)
			{
				in_XX.push_back(xx);
				in_YY.push_back(yy);
				in_ZZ.push_back(zz);
				in_XY.push_back(xy);
				in_XZ.push_back(xz);
				in_YZ.push_back(yz);

				num_points_read_in++;
			}
		
			infile.close();
		}

		void splinify(uint32_t num_points)
		{
			if(num_points_read_in == 0) {
				fprintf(stderr, "Error: No strain data has been read in yet. Please use .from_file() first.\n");
				exit(1);
			}

			this->num_points = num_points;

			tk::spline splXX, splYY, splZZ, splXY, splXZ, splYZ;

			// Set splines
			std::vector<double> T;
			for(uint32_t n = 0; n < num_points_read_in; n++) {
				double t = (double)n/(double)(num_points_read_in - 1);
				T.push_back(t);
			}

			splXX.set_points(T,in_XX);
			splYY.set_points(T,in_YY);
			splZZ.set_points(T,in_ZZ);
			splXY.set_points(T,in_XY);
			splXZ.set_points(T,in_XZ);
			splYZ.set_points(T,in_YZ);

			for(uint32_t n = 0; n < num_points; n++) {
				double t = (double)n/(double)(num_points - 1);
				XX.push_back(splXX(t));
				YY.push_back(splYY(t));
				ZZ.push_back(splZZ(t));
				XY.push_back(splXY(t));
				XZ.push_back(splXZ(t));
				YZ.push_back(splYZ(t));
			}
		}

		void to_file(char *out_fname)
		{
			std::ofstream outfile(out_fname);
			if(outfile.fail()) {
				fprintf(stderr, "Could not open %s for writing.\n", out_fname);
				exit(1);
			}

			for(uint32_t n = 0; n < num_points; n++) {
			outfile << XX[n] << ' ' << YY[n] << ' ' << ZZ[n] << ' ' << XY[n] << ' ' << XZ[n] << ' ' << YZ[n] << '\n';
			}

			outfile.close();
		}

	private:
		uint32_t num_points_read_in;
		std::vector<double> in_XX, in_YY, in_ZZ, in_XY, in_XZ, in_YZ;
};

double compare_L2_norm(Strain6D *a, Strain6D *b)
{
	if(a->num_points != b->num_points) {
		fprintf(stderr, "Error in compare_L2_norm(): given strain6D objects have different numbers of spline points (%u and %u)\n", a->num_points, b->num_points);
		exit(1);
	}

	uint32_t N = a->num_points;
	double sum = 0;
	for(uint32_t i = 0; i < N; i++) {
		double dxx = a->XX[i] - b->XX[i];
		double dyy = a->YY[i] - b->YY[i];
		double dzz = a->ZZ[i] - b->ZZ[i];
		double dxy = a->XY[i] - b->XY[i];
		double dxz = a->XZ[i] - b->XZ[i];
		double dyz = a->YZ[i] - b->YZ[i];

		sum += dxx*dxx + dyy*dyy + dzz*dzz + dxy*dxy + dxz*dxz + dyz*dyz;
	}

	return sqrt(sum);
}

#endif /* MATHISTPREDICT_STRAIN2SPLINE_H */

