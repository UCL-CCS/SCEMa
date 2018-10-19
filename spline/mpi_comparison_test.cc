#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include <stdint.h>
#include <vector>
#include <iostream>
#include <limits.h>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>


#include <mpi.h>

#include "strain2spline.h"

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

void erase_substring(std::string & mainStr, const std::string &toErase)
{
	size_t pos = mainStr.find(toErase);
	if (pos != std::string::npos) {
		mainStr.erase(pos, toErase.length());
	}
}

int main(int argc, char **argv)
{
	if(argc != 4) {
		fprintf(stderr, "Usage: ./mpi_comparison_test STRAIN_DIRECTORY NUM_SPLINE_POINTS THRESH\n");
		return 1;
	}

	char *straindir = argv[1];
	uint32_t num_points = atoi(argv[2]);
	double acceptable_diff_threshold = atof(argv[3]); 

	MPI_Comm comm = MPI_COMM_WORLD;

	// set up MPI
	MPI_Init(NULL, NULL);

	int32_t this_rank, num_ranks;
	MPI_Comm_rank(comm, &this_rank);
	MPI_Comm_size(comm, &num_ranks);

	// Get list of strain history files in given directory
	std::vector<std::string> fnames;
	read_directory(straindir, fnames);

	// Read all strain histories to vector
	//time_t start_read = time(NULL);
	std::vector<MatHistPredict::Strain6D*> histories;
	int32_t i = 0;
	for(std::string fname : fnames) {
		if(!starts_with(fname, "strain_")) {
			std::cout << "Ignoring: '" << fname << "'\n";
			continue;
		}

		if(i % num_ranks == this_rank) {
			MatHistPredict::Strain6D *new_s6D = new MatHistPredict::Strain6D();
			new_s6D->from_file((std::string(straindir) + fname).c_str());

			new_s6D->splinify(num_points);

			erase_substring(fname, "strain_");
			uint32_t id = atoi(fname.c_str());
//			std::cout << id << "\n";
			new_s6D->set_ID(id);

			histories.push_back(new_s6D);
		}
		i++;
	}
	uint32_t num_histories_on_this_rank = histories.size();
	//time_t end_read = time(NULL);

	
	// Find the most similar strain histories
	MatHistPredict::compare_histories_with_all_ranks(histories, acceptable_diff_threshold, comm);

	// Results
	for(uint32_t i=0; i < num_histories_on_this_rank; i++) {
		//bool will_run_new_MD = histories[i]->run_new_sim(acceptable_diff_threshold);
//		std::cout << "Rank " << this_rank << ": Hist" << histories[i]->get_ID() << " is most similar to Hist" << histories[i]->get_most_similar_history_ID() << " with diff " << histories[i]->get_most_similar_history_diff() << " => will_run_new_MD=" << will_run_new_MD << "\n";

		//histories[i]->print_most_similar_histories();
		histories[i]->most_similar_histories_to_file(("__results/ID_" + std::to_string(histories[i]->get_ID()) + ".txt").c_str());
	}

	MPI_Finalize();

	return 0;
}
