#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>
 
#include "../headers/strain2spline.h"

bool starts_with(std::string mainStr, std::string toMatch)
{
	// std::string::find returns 0 if toMatch is found at starting
	if(mainStr.find(toMatch) == 0)
		return true;
	else
		return false;
}

void read_directory(const std::string& name, std::vector<std::string>& v)
{
	DIR* dirp = opendir(name.c_str());
	struct dirent * dp;
	while ((dp = readdir(dirp)) != NULL) {
		v.push_back(dp->d_name);
	}
	closedir(dirp);
}

int main(int argc, char** argv) {

	if(argc != 3) {
		fprintf(stderr, "Usage: ./compare_all_histories STRAIN_DIRECTORY NUM_SPLINE_POINTS\n");
		return 1;
	}
	char *straindir = argv[1];
	uint32_t num_points = atoi(argv[2]);

	// Get list of strain history files in given directory
	std::vector<std::string> fnames;
	read_directory(straindir, fnames);

	// Read all strain histories to vector
	time_t start_read = time(NULL);
	std::vector<Strain6D*> strains;
	for(std::string fname : fnames) {
		if(!starts_with(fname, "strain_")) {
			std::cout << "Ignoring: '" << fname << "'\n";
			continue;
		}
		std::cout << "Reading: '" << fname << "'\n";
		Strain6D *s6D = new Strain6D();
		s6D->from_file((std::string(straindir) + fname).c_str());

		strains.push_back(s6D);
	}
	time_t end_read = time(NULL);

	// Splinify all histories
	time_t start_spline = time(NULL);
	uint32_t num_cells = strains.size();
	for(uint32_t i = 0; i < num_cells; i++) {
		strains[i]->splinify(num_points);
	}
	time_t end_spline = time(NULL);

	// Carry out all comparisons
	time_t start_compare = time(NULL);
	for(uint32_t i = 0; i < num_cells; i++) {
		for(uint32_t j = i; j < num_cells; j++) {
			double L2 = compare_L2_norm(strains[i], strains[j]);
			std::cout << strains[i]->in_fname << " vs " << strains[j]->in_fname << ":" << L2 << "\n";
		}
	}
	time_t end_compare = time(NULL);

	double read_time = double(end_read - start_read);
	double spline_time = double(end_spline - start_spline);
	double compare_time = double(end_compare - start_compare);

	std::cout << "Read time: " << read_time << "\n";
	std::cout << "Spline time: " << spline_time << "\n";
	std::cout << "Compare time: " << compare_time << "\n";

	return 0;
}
