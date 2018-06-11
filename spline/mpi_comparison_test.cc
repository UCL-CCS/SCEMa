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

// Handle negative numbers too
int32_t modulo_neg(int32_t x, int32_t n)
{
	return (x % n + n) % n;
}

typedef struct
{
	uint32_t num_histories;
	std::vector<Strain6D*> histories;
} RANKDATA;

int main(int argc, char **argv)
{
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Status status;
	MPI_Request request;

	// set up MPI
	MPI_Init(NULL, NULL);

	int32_t this_rank, num_ranks;
	MPI_Comm_rank(comm, &this_rank);
	MPI_Comm_size(comm, &num_ranks);

	// Test data on this rank
	RANKDATA rankdata;
	rankdata.num_histories = 1;
	for(uint32_t i = 0; i < rankdata.num_histories; i++) {
		Strain6D *new_hist = new Strain6D();
		for(uint32_t j = 0; j < 10; j++) {
			new_hist->add_current_strain(this_rank, this_rank, this_rank, this_rank, this_rank, this_rank); // for testing
		}
		new_hist->splinify(10);
		new_hist->print();
		rankdata.histories.push_back(new_hist);
	}

	int32_t recv_max_buf_size = 100;
	uint32_t recv_num_histories = 0;
	uint32_t recv_num_hist_points = 0;
	double recv_buf[recv_max_buf_size];
	for(int32_t i = 1; i < num_ranks; i++) {
		int32_t target_rank = modulo_neg(this_rank + i, num_ranks);
		int32_t from_rank = modulo_neg(this_rank - i, num_ranks);
		
		std::cout << "Rank " << this_rank << ": Targetting " << target_rank << "\n";
		MPI_Isend(&(rankdata.num_histories), 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);
		for(uint32_t j = 0; j < rankdata.num_histories; j++) {
			uint32_t num_hist_points = rankdata.histories[j]->num_points;
			MPI_Isend(&(num_hist_points), 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);
			MPI_Isend(&(rankdata.histories[j]->spline[0]), num_hist_points, MPI_DOUBLE, target_rank, this_rank, comm, &request);
		}

		std::cout << "Rank " << this_rank << ": Expecting " << from_rank << "\n";
		MPI_Recv(&recv_num_histories, 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);
		for(uint32_t j = 0; j < recv_num_histories; j++) {
			MPI_Recv(&recv_num_hist_points, 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);
			MPI_Recv(recv_buf, recv_max_buf_size, MPI_DOUBLE, from_rank, from_rank, comm, &status);

			for(uint32_t k = 0; k < recv_num_hist_points; k++) {
				std::cout << "Received from rank " << from_rank << " " << recv_buf[j] << "\n";
			}
		}
	}

	MPI_Finalize();

	return 0;
}
