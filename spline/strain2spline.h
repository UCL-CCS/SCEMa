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
//		std::vector<double> XX, YY, ZZ, XY, XZ, YZ;
//		std::string in_fname;
		std::vector<double> spline;

		Strain6D()
		{
			num_points_read_in = 0;
			num_points = 0;
		}

		void add_current_strain(double xx, double yy, double zz, double xy, double xz, double yz)
		{
			in_XX.push_back(xx);
			in_YY.push_back(yy);
			in_ZZ.push_back(zz);
			in_XY.push_back(xy);
			in_XZ.push_back(xz);
			in_YZ.push_back(yz);
			num_points_read_in++;
		}

		void from_file(const char *in_fname)
		{
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
				fprintf(stderr, "Error: No strain data has been read in yet. Please use .from_file() or .add_current_strain() first.\n");
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

			std::cout << T.size() << " " << in_XX.size() << "\n";

			splXX.set_points(T,in_XX);
			splYY.set_points(T,in_YY);
			splZZ.set_points(T,in_ZZ);
			splXY.set_points(T,in_XY);
			splXZ.set_points(T,in_XZ);
			splYZ.set_points(T,in_YZ);

			spline.clear(); // reset the existing spline result to zero
			spline.reserve(this->num_points * 6);
			for(uint32_t n = 0; n < num_points; n++) {
				double t = (double)n/(double)(num_points - 1);
				spline.push_back(splXX(t));
				spline.push_back(splYY(t));
				spline.push_back(splZZ(t));
				spline.push_back(splXY(t));
				spline.push_back(splXZ(t));
				spline.push_back(splYZ(t));
			}
		}

		void print()
		{
			for(uint32_t n = 0; n < num_points * 6; n += 6) {
				std::cout << spline[n] << ' ' << spline[n + 1] << ' ' << spline[n + 2] << ' ' << spline[n + 3] << ' ' << spline[n + 4] << ' ' << spline[n + 5] << '\n';
			}
		}

		void to_file(char *out_fname)
		{
			std::ofstream outfile(out_fname);
			if(outfile.fail()) {
				fprintf(stderr, "Could not open %s for writing.\n", out_fname);
				exit(1);
			}

			for(uint32_t n = 0; n < num_points * 6; n += 6) {
				outfile << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << '\n';
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
		double diff = a->spline[i] - b->spline[i];
		sum += diff*diff;
	}

	return sqrt(sum);
}

double compare_L2_norm(double *a, double *b, uint32_t num_points_a, uint32_t num_points_b)
{
	if(num_points_a != num_points_b) {
		fprintf(stderr, "Error in compare_L2_norm(): given strain6D objects have different numbers of spline points (%u and %u)\n", num_points_a, num_points_b);
		exit(1);
	}

	uint32_t N = num_points_a;
	double sum = 0;
	for(uint32_t i = 0; i < N; i++) {
		double diff = a[i] - b[i];
		sum += diff*diff;
	}

	return sqrt(sum);
}

#endif /* MATHISTPREDICT_STRAIN2SPLINE_H */

