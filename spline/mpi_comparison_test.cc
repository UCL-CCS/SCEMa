#include <stdio.h>
#include <cstring>
#include <string>
#include <stdint.h>
#include <math.h>
#include <rpc/rpc.h>
#include <zlib.h>
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include <limits.h>
#include <vector>
#include <assert.h>
#include <mpi.h>

#include "strain2spline.h"

int main(int argc, char **argv)
{
	MPI_Comm comm = MPI_COMM_WORLD;

	// set up MPI
	MPI_Init(NULL, NULL);

	int32_t this_rank, num_ranks;
	MPI_Comm_rank(comm, &this_rank);
	MPI_Comm_size(comm, &num_ranks);

	// Build some Strain6D objects for this rank
	uint32_t num_histories_on_this_rank = 3;
	std::vector<MatHistPredict::Strain6D*> histories;
	for(uint32_t i = 0; i < num_histories_on_this_rank; i++) {
		MatHistPredict::Strain6D *new_s6D = new MatHistPredict::Strain6D();

		for(uint32_t j = 0; j < 5; j++) {
			new_s6D->add_current_strain(	this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i,
							this_rank * 10 + i);
		}
		new_s6D->splinify(10);
		new_s6D->set_ID(this_rank * 10 + i);
		new_s6D->print();
		histories.push_back(new_s6D);
	}
	
	// Find the most similar strain histories
	MatHistPredict::compare_histories_with_all_ranks(histories, comm);

	// Results
	for(uint32_t i=0; i < num_histories_on_this_rank; i++) {
		std::cout << "Rank " << this_rank << ": History " << histories[i]->get_ID() << " is most similar to history " << histories[i]->get_most_similar_history_ID() << " with diff " << histories[i]->get_most_similar_history_diff() << "\n";
	}

	MPI_Finalize();

	return 0;
}
