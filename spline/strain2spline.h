#ifndef MATHISTPREDICT_STRAIN2SPLINE_H
#define MATHISTPREDICT_STRAIN2SPLINE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>

#include "spline.h"

typedef struct
{
	std::pair <uint32_t, uint32_t> ID;
	double diff;
} HISTORY_ID_DIFF_PAIR;

namespace MatHistPredict {

	class Strain6D
	{
		public:
			Strain6D()
			{
				up_to_date = false;
				num_steps_added = 0;
				num_spline_points_per_component = 0;

				ID = std::pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(),
									 	 	 	   std::numeric_limits<uint32_t>::max()); // Should be set correctly using set_ID
				ID_is_set = false;

				most_similar_history.ID = std::pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(),
							 	 	 	 	 	 	 	 	 	 	 	std::numeric_limits<uint32_t>::max());
				most_similar_history.diff = 0;

				most_similar_histories.clear();

				ID_to_get_results_from = ID = std::pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(),
		 	 	 	    													std::numeric_limits<uint32_t>::max());
			}

			void set_ID(std::pair <uint32_t, uint32_t> ID)
			{
				this->ID = ID;
				this->ID_is_set = true;
			}

			void add_current_strain(double strain_xx, double strain_yy, double strain_zz, double strain_xy, double strain_xz, double strain_yz)
			{
				up_to_date = false;

				in_XX.push_back(strain_xx);
				in_YY.push_back(strain_yy);
				in_ZZ.push_back(strain_zz);
				in_XY.push_back(strain_xy);
				in_XZ.push_back(strain_xz);
				in_YZ.push_back(strain_yz);
				num_steps_added++;
			}

			void add_current_strain(double strain_xx, double strain_yy, double strain_zz, double strain_xy, double strain_xz, double strain_yz,
						double stress_xx, double stress_yy, double stress_zz, double stress_xy, double stress_xz, double stress_yz)
			{
				up_to_date = false;

				in_XX.push_back(strain_xx);
				in_YY.push_back(strain_yy);
				in_ZZ.push_back(strain_zz);
				in_XY.push_back(strain_xy);
				in_XZ.push_back(strain_xz);
				in_YZ.push_back(strain_yz);
				num_steps_added++;

				// Also keep track of most recent stress
				this->stress[0] = stress_xx;
				this->stress[1] = stress_yy;
				this->stress[2] = stress_zz;
				this->stress[3] = stress_xy;
				this->stress[4] = stress_xz;
				this->stress[5] = stress_yz;
			}

			/* Read in strain values from file in_fname */
			void from_file(const char *in_fname)
			{
				up_to_date = false;

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

					num_steps_added++;
				}
				infile.close();
			}

			/* Build a spline out of the strain steps that have been read-in so far (must be at least 3 steps).
			 * Each component is represented by num_spline_points_per_component equally spaced points along the
			 * spline. The total number of points in the final strain vector is therefore num_spline_points_per_component * 6.
			 */
			void splinify(uint32_t num_spline_points_per_component)
			{
				if(num_steps_added == 0) {
					fprintf(stderr, "Error: Nothing to splinify! No strain data has been read in yet. Please use .from_file() or .add_current_strain() first.\n");
					exit(1);
				} else if(num_steps_added < 3) {
					fprintf(stderr, "Error: Not enough strain steps added. Need at least 3 points for splinify().\n");
					exit(1);
				}

				this->num_spline_points_per_component = num_spline_points_per_component;

				tk::spline splXX, splYY, splZZ, splXY, splXZ, splYZ;

				// Set splines
				std::vector<double> T;
				for(uint32_t n = 0; n < num_steps_added; n++) {
					double t = (double)n/(double)(num_steps_added - 1);
					T.push_back(t);
				}

				splXX.set_points(T,in_XX);
				splYY.set_points(T,in_YY);
				splZZ.set_points(T,in_ZZ);
				splXY.set_points(T,in_XY);
				splXZ.set_points(T,in_XZ);
				splYZ.set_points(T,in_YZ);

				spline.clear(); // reset the existing spline result to zero
				spline.reserve(num_spline_points_per_component * 6); // mult by 6 because there are 6 components
				for(uint32_t n = 0; n < num_spline_points_per_component; n++) {
					double t = (double)n/(double)(num_spline_points_per_component - 1);
					spline.push_back(splXX(t));
					spline.push_back(splYY(t));
					spline.push_back(splZZ(t));
					spline.push_back(splXY(t));
					spline.push_back(splXZ(t));
					spline.push_back(splYZ(t));
				}

				up_to_date = true;
			}

			void print()
			{
				if(!up_to_date) {
					std::cout << "Warning: spline is not up to date (run splinify() to rebuild spline)\n";
				}
				for(uint32_t n = 0; n < num_spline_points_per_component * 6; n += 6) {
					std::cout << spline[n] << ' ' << spline[n + 1] << ' ' << spline[n + 2] << ' ' << spline[n + 3] << ' ' << spline[n + 4] << ' ' << spline[n + 5] << '\n';
				}
			}

			void spline_to_file(char *out_fname)
			{
				if(!up_to_date) {
					std::cout << "Warning: spline is not up to date (run splinify() to rebuild spline)\n";
				}

				std::ofstream outfile(out_fname);
				if(outfile.fail()) {
					fprintf(stderr, "Could not open %s for writing.\n", out_fname);
					exit(1);
				}

				for(uint32_t n = 0; n < num_spline_points_per_component * 6; n += 6) {
					outfile << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << ' ' << spline[n] << '\n';
				}

				outfile.close();
			}

			std::vector<double> * get_spline()
			{
				return &spline;
			}

			std::pair <uint32_t, uint32_t>& get_ID()
			{
				if(!ID_is_set) {
					fprintf(stderr, "Error: history ID is unset. Please use set_ID().\n");
					exit(1);
				}
				return ID;
			}

			uint32_t get_num_spline_points_per_component()
			{
				return num_spline_points_per_component;
			}

			std::pair <uint32_t, uint32_t> get_most_similar_history_ID()
			{
				return most_similar_history.ID;
			}

			double get_most_similar_history_diff()
			{
				return most_similar_history.diff;
			}
			
			void clear_most_similar_history()
			{
				most_similar_history.ID = std::pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(),
		 	 	 	    												std::numeric_limits<uint32_t>::max());
				most_similar_history.diff = std::numeric_limits<double>::infinity();

				most_similar_histories.clear();
				all_similar_histories.clear();
			}

			void choose_most_similar_history(double candidate_diff, std::pair <uint32_t, uint32_t>& candidate_ID, double threshold)
			{
				HISTORY_ID_DIFF_PAIR hp;
				hp.diff = candidate_diff;
				hp.ID = candidate_ID;

				all_similar_histories.push_back(hp);
				if(candidate_diff < threshold) {
					/*HISTORY_ID_DIFF_PAIR hp;
					hp.diff = candidate_diff;
					hp.ID = candidate_ID;*/
					most_similar_histories.push_back(hp);
				}

				if(candidate_diff <= most_similar_history.diff) {

					// In the rare case where several histories may be exactly equidistant,
					// choose the candidate_ID that is lowest
					if(candidate_diff == most_similar_history.diff) {
						if(candidate_ID.first > most_similar_history.ID.first)
							if(candidate_ID.second > most_similar_history.ID.second){
								return; // reject
							}
					}

					most_similar_history.ID = candidate_ID;
					most_similar_history.diff = candidate_diff;
				}
			}

			void print_most_similar_histories()
			{
				for(uint32_t i = 0; i < most_similar_histories.size(); i++) {
					std::cout << ID.first << "-" << ID.second << " " << most_similar_histories[i].ID.first << "-" << most_similar_histories[i].ID.second << " " << most_similar_histories[i].diff << "\n";
				}
			}

			void most_similar_histories_to_file(const char *out_fname)
			{
				std::ofstream outfile(out_fname);
				if(outfile.fail()) {
					fprintf(stderr, "Could not open %s for writing.\n", out_fname);
					exit(1);
				}

				for(uint32_t i = 0; i < most_similar_histories.size(); i++) {
					outfile << ID.first << "-" << ID.second << " " << most_similar_histories[i].ID.first << "-" << most_similar_histories[i].ID.second << " " << most_similar_histories[i].diff << "\n";
				}

				outfile.close();
			}

			void all_similar_histories_to_file(const char *out_fname)
			{
				std::ofstream outfile(out_fname);
				if(outfile.fail()) {
					fprintf(stderr, "Could not open %s for writing.\n", out_fname);
					exit(1);
				}

				for(uint32_t i = 0; i < all_similar_histories.size(); i++) {
					outfile << ID.first << "-" << ID.second << " " << all_similar_histories[i].ID.first << "-" << all_similar_histories[i].ID.second << " " << all_similar_histories[i].diff << "\n";
				}

				outfile.close();
			}

			bool run_new_md()
			{
				if(ID_to_get_results_from == ID) {
					return true;
				}
				return false;
			}

			/* Read mapping.csv file output from the coarsegrain_dependency_network.py script */
			void read_coarsegrain_dependency_mapping(const char *in_fname)
			{
				std::ifstream infile(in_fname);
				if(infile.fail()) {
					fprintf(stderr, "Could not open %s for reading.\n", in_fname);
					exit(1);
				}
				char id_from[1024], id_to[1024];

				// Skip lines until line that contains this ID
				// Line number doesn't not always fit cell ID because
				// if cell has no similar history it is not captured by the python script
				std::string line;
				for(uint32_t i = 0; i < ID.first; i++)
					for(uint32_t i = 0; i < ID.second; i++) {
						std::getline(infile, line);
					}
				infile >> id_from >> id_to;

				//std::cout << "cell " << ID << "id_from " << id_from << "id_to" << id_to << std::endl;
				char qcid[1024]; sprintf(qcid, "%d-%d", ID.first, ID.second);
				if(id_from != qcid) {
					fprintf(stderr, "ID in mapping file (%s) does not match cell ID (%s)\n", id_from, qcid);
					exit(1);
				}

				std::istringstream ss(id_to);
				std::string token;
				std::vector<int> tmp;

				while(std::getline(ss, token, '-')) {
				    tmp.push_back(std::stoi(token));
				}

				this->ID_to_get_results_from = std::pair<uint32_t, uint32_t>(tmp[0], tmp[1]);

				infile.close();
			}

			void set_ID_to_get_results_from(std::pair <uint32_t, uint32_t> ID)
			{
				this->ID_to_get_results_from = ID;
			}

			std::pair <uint32_t, uint32_t> get_ID_to_update_from()
			{
				return ID_to_get_results_from;
			}

		private:
			bool up_to_date;

			uint32_t num_steps_added;
			std::vector<double> in_XX, in_YY, in_ZZ, in_XY, in_XZ, in_YZ; // input strain at each timestep (used to build spline)

			// Stress at most recent step
			double stress[6];

			// Integer ID of the cell/quad point that this strain history belongs to
			bool ID_is_set;
			std::pair <uint32_t, uint32_t> ID;

			uint32_t num_spline_points_per_component;
			std::vector<double> spline; // built spline

			// The ID and L2 norm difference of the most similar strain history, calculated by running compare_histories_with_all_ranks()
			HISTORY_ID_DIFF_PAIR most_similar_history;

			// List of all (other) histories within threshold difference of this history			
			std::vector<HISTORY_ID_DIFF_PAIR> most_similar_histories;
			std::vector<HISTORY_ID_DIFF_PAIR> all_similar_histories;

			std::pair <uint32_t, uint32_t> ID_to_get_results_from;
	};

	class Strain6DReceiver
	{
		public:
			Strain6DReceiver(int32_t max_buf_size)
			{
				this->max_buf_size = max_buf_size;
				spline = new double[max_buf_size];
			}

			~Strain6DReceiver()
			{
				delete[] spline;
				this->max_buf_size = 0;
			}

			int32_t max_buf_size; // Max number of doubles that can be received from another rank
			double *spline;
			std::pair <uint32_t, uint32_t>  ID;
			uint32_t recv_count;
	};

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

	double compare_L2_norm(Strain6D *a, Strain6D *b)
	{
		double *hist_A = a->get_spline()->data();
		double *hist_B = b->get_spline()->data();
		uint32_t num_points_A = a->get_spline()->size();
		uint32_t num_points_B = b->get_spline()->size();
		return compare_L2_norm(hist_A, hist_B, num_points_A, num_points_B);
	}

	double compare_L2_norm(Strain6D *a, Strain6DReceiver *b)
	{
		double *hist_A = a->get_spline()->data();
		double *hist_B = b->spline;
		uint32_t num_points_A = a->get_spline()->size();
		uint32_t num_points_B = b->recv_count;
		return compare_L2_norm(hist_A, hist_B, num_points_A, num_points_B);
	}

	void send_strain6D_mpi(Strain6D *in_s6D, int32_t target_rank, int32_t this_rank, MPI_Comm comm)
	{
		MPI_Request request;

		std::vector<double> *strain = in_s6D->get_spline();
		std::pair <uint32_t, uint32_t> ID = in_s6D->get_ID();
		int32_t num_doubles_to_send = strain->size();

		MPI_Isend(&num_doubles_to_send, 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);
		MPI_Isend(strain->data(), num_doubles_to_send, MPI_DOUBLE, target_rank, this_rank, comm, &request);
		MPI_Isend(&ID, 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);
	}

	void receive_strain6D_mpi(Strain6DReceiver *recv, int32_t from_rank, MPI_Comm comm)
	{
		MPI_Status status;
		MPI_Recv(&(recv->recv_count), 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);
		MPI_Recv(recv->spline, recv->max_buf_size, MPI_DOUBLE, from_rank, from_rank, comm, &status);
		MPI_Recv(&(recv->ID), 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);
	}

	// Handle negative numbers too
	int32_t modulo_neg(int32_t x, int32_t n)
	{
		return ((x%n + n) % n);
	}

	void compare_histories_with_all_ranks(std::vector<Strain6D*>& histories, double threshold, MPI_Comm comm)
	{
		MPI_Request request;
		MPI_Status status;

		int32_t this_rank, num_ranks;
		MPI_Comm_rank(comm, &this_rank);
		MPI_Comm_size(comm, &num_ranks);
		
		// For receiving the strain history and ID from another rank
		const int32_t max_buf_size = 2000;
		Strain6DReceiver recv(max_buf_size);

		// num strain6D histories on this rank
		uint32_t num_histories_on_this_rank = histories.size();

		// reset the most similar history vars for each Strain6D to the maximum difference
		// (in preparation for minimisation)
		for(uint32_t h = 0; h < num_histories_on_this_rank; h++) {
			histories[h]->clear_most_similar_history();
		}

		// Cycle through all ranks in the communicator in a ring-like fashion, sending
		// to this_rank+i (periodic) and receiving from this_rank-i (periodic). This
		// ensures that every rank gets the data from every other rank (for comparison)
		// wihout ever needing to hold all cells in memory at once.
		for(int32_t i = 0; i < num_ranks; i++) {
			int32_t target_rank = modulo_neg(this_rank + i, num_ranks); // send data to target_rank
			int32_t from_rank = modulo_neg(this_rank - i, num_ranks); // receive data sent by from_rank
			
			// If we are not considering cells on the same rank
//			std::cout << "Rank " << this_rank << ": Targetting " << target_rank << " Expecting " << from_rank << "\n";
			if(target_rank != this_rank) {
				// Indicate the number of histories that will be sent to target_rank
				MPI_Isend(&num_histories_on_this_rank, 1, MPI_UNSIGNED, target_rank, this_rank, comm, &request);

				// Send all histories and IDs
				for(uint32_t h = 0; h < num_histories_on_this_rank; h++) {
					send_strain6D_mpi(histories[h], target_rank, this_rank, comm);
				}

				// Get number of histories to receive from from_rank
				uint32_t num_histories_to_receive = 0;
				MPI_Recv(&num_histories_to_receive, 1, MPI_UNSIGNED, from_rank, from_rank, comm, &status);

				// Receive histories from rank from_rank, and compare with all histories on this rank
				for(uint32_t r = 0; r < num_histories_to_receive; r++) {

					receive_strain6D_mpi(&recv, from_rank, comm);

					for(uint32_t h = 0; h < num_histories_on_this_rank; h++) {
						double diff = compare_L2_norm(histories[h], &recv);
						histories[h]->choose_most_similar_history(diff, recv.ID, threshold);
//						std::cout << "Comparison between rank " << this_rank << ", cell " << histories[h]->get_ID() << " and rank " <<  from_rank << ", cell " << recv.ID << ": " << diff << "\n";
					}
				}

			} else { // Considering cells on the same rank
				for(uint32_t a = 0; a < num_histories_on_this_rank; a++) {
					for(uint32_t b = a + 1; b < num_histories_on_this_rank; b++) {
						double diff = compare_L2_norm(histories[a], histories[b]);
						histories[a]->choose_most_similar_history(diff, histories[b]->get_ID(), threshold); // both Strain6D's need this info
						histories[b]->choose_most_similar_history(diff, histories[a]->get_ID(), threshold); // both Strain6D's need this info
						//std::cout << "Same rank comparison (" << this_rank << ") cell " << histories[a]->get_ID() << " vs cell " << histories[b]->get_ID() << ": " << diff << "\n";
					}
				}
			}
		}
	}
}
#endif /* MATHISTPREDICT_STRAIN2SPLINE_H */

