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

#include <mpi.h>

#include "strain2spline.h"

int main(int argc, char **argv)
{
	const double acceptable_diff_threshold = 100.0; 

	MPI_Comm comm = MPI_COMM_WORLD;

	// set up MPI
	MPI_Init(NULL, NULL);

	int32_t this_rank, num_ranks;
	MPI_Comm_rank(comm, &this_rank);
	MPI_Comm_size(comm, &num_ranks);

	// Build some Strain6D objects for this rank
	uint32_t num_histories_on_this_rank = 1;
	std::vector<MatHistPredict::Strain6D*> histories;
	std::srand(this_rank*100); // never use srand(0) or srand(1) - they give the same...
	std::cout << "Rank " << this_rank << ": " << std::rand() << "\n";
	for(uint32_t i = 0; i < num_histories_on_this_rank; i++) {
		MatHistPredict::Strain6D *new_s6D = new MatHistPredict::Strain6D();

		for(uint32_t j = 0; j < 5; j++) {
			new_s6D->add_current_strain(	std::rand()/(float)RAND_MAX,
							std::rand()/(float)RAND_MAX,
							std::rand()/(float)RAND_MAX,
							std::rand()/(float)RAND_MAX,
							std::rand()/(float)RAND_MAX,
							std::rand()/(float)RAND_MAX);
		}
		new_s6D->splinify(10);
		new_s6D->set_ID(this_rank * 10 + i);
		std::cout << "Rank " << this_rank << ":\n";
		new_s6D->print();
		histories.push_back(new_s6D);
	}
	
	// Find the most similar strain histories
	MatHistPredict::compare_histories_with_all_ranks(histories, acceptable_diff_threshold, comm);

	// Results
	for(uint32_t i=0; i < num_histories_on_this_rank; i++) {
		bool will_run_new_MD = histories[i]->run_new_sim(acceptable_diff_threshold);
		std::cout << "Rank " << this_rank << ": Hist" << histories[i]->get_ID() << " is most similar to Hist" << histories[i]->get_most_similar_history_ID() << " with diff " << histories[i]->get_most_similar_history_diff() << " => will_run_new_MD=" << will_run_new_MD << "\n";

		histories[i]->print_most_similar_histories();
		histories[i]->most_similar_histories_to_file(("Rank_" + std::to_string(this_rank) + ".txt").c_str());
	}

	MPI_Finalize();

	return 0;
}
