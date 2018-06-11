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

// Handle negative numbers too
int32_t modulo(int32_t x, int32_t n)
{
	return (x % n + n) % n;
}

typedef struct
{
	uint32_t num_cells;
	std::vector<uint32_t> cells;
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

	// Data on this rank
	RANKDATA rankdata;
	rankdata.num_cells = 3;
	for(uint32_t i = 0; i < rankdata.num_cells; i++) {
		rankdata.cells.push_back(this_rank * 10 + i);
	}

	int32_t recv_buf_size = 10;
	uint32_t recv_buf_count = 0;
	uint32_t recv_buf[recv_buf_size];
	for(int32_t i = 1; i < num_ranks; i++) {
		int32_t target_rank = modulo(this_rank + i, num_ranks);
		int32_t from_rank = modulo(this_rank - i, num_ranks);
		
		std::cout << "Rank " << this_rank << ": Targetting " << target_rank << "\n";
		MPI_Isend(&(rankdata.num_cells), 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);
		MPI_Isend(&(rankdata.cells[0]), rankdata.cells.size(), MPI_UNSIGNED, target_rank, this_rank, comm, &request);

		std::cout << "Rank " << this_rank << ": Expecting " << from_rank << "\n";
		MPI_Recv(&recv_buf_count, 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);
		MPI_Recv(recv_buf, recv_buf_size, MPI_UNSIGNED, from_rank, from_rank, comm, &status);

		for(uint32_t j = 0; j < recv_buf_count; j++) {
			std::cout << "Received from rank " << from_rank << " " << recv_buf[j] << "\n";
		}
	}


/*
	MPI_Allreduce(local_block_dims, global_block_dims, 3, MPI_UNSIGNED_LONG, MPI_MAX, comm);
*/

/*
	// get receive counts from each rank
	int *recvcounts = NULL;
	if(this_rank == 0) {
		recvcounts = new int[num_ranks];
	}
	int num_lints_to_send = records->size();
	MPI_Gather(&num_lints_to_send, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);

	// prepare the arrays rank 0 will need to receive all other ranks' records (from current chunk)
	int *displs = NULL;
	int64_t *all_records = NULL;
	uint64_t total_num_lints = 0;
	if(this_rank == 0) {
		displs = new int[num_ranks];
		for(int j = 0; j < num_ranks; j++) {
			displs[j] = total_num_lints;
			total_num_lints += recvcounts[j];
		}
		all_records = new int64_t[total_num_lints];
	}
		int64_t *to_send = NULL;
	if(num_lints_to_send > 0) {
		to_send = &(*records)[0]; // this looks so awful...
	}
		if(this_rank == 0) {
		DEBUGMSG("%u] Require %lu records. Break-down:\n", i, total_num_lints/section->get_num_lints_per_record());
		for(int j = 0; j < num_ranks; j++) {
			DEBUGMSG("%u] Rank %d will send %ld\n", i, j, recvcounts[j]/section->get_num_lints_per_record());
		}
	}

	// collect all relevant records (in this chunk) from all ranks
	MPI_Gatherv(to_send, num_lints_to_send, MPI_LONG, all_records, recvcounts, displs, MPI_LONG, 0, comm);
*/

	MPI_Finalize();

	return 0;
}
