#ifndef MMD_PROBLEM_H
#define MMD_PROBLEM_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "read_write.h"
#include "tensor_calc.h"
#include "init_material_problem.h"

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/mpi.h>

namespace HMM
{
	using namespace dealii;

	template <int dim>
	struct ReplicaData
	{
		// Characteristics
		double rho;
		std::string mat;
		int repl;
		int nflakes;
		Tensor<1,dim> init_length;
		Tensor<2,dim> rotam;
		SymmetricTensor<2,dim> init_stress;
		SymmetricTensor<4,dim> init_stiff;
	};



	template <int dim>
	class EQMDSync
	{
	public:
		EQMDSync (MPI_Comm mcomm, int pcolor);
		~EQMDSync ();

		void equilibrate (double mdtlength, double mdtemp, int nss, int nse, double strr, double stra,
				   std::string ffi, std::string nslocin, std::string nlogloc,
				   std::string nlogloctmp,
				   std::string mdsdir, unsigned int bnmin, unsigned int mppn,
				   std::vector<std::string> mdt, Tensor<1,dim> cgd, unsigned int nr, bool ups);

	private:

		void set_md_procs (int nmdruns);

		void load_replica_generation_data();

		void prepare_replica_equilibration();
		void equilibrate_replicas ();

		MPI_Comm 							mmd_communicator;
		MPI_Comm 							md_batch_communicator;
		int 								mmd_n_processes;
		int 								md_batch_n_processes;
		int 								n_md_batches;
		int 								this_mmd_process;
		int 								this_md_batch_process;
		int 								mmd_pcolor;
		int									md_batch_pcolor;

		unsigned int						machine_ppn;
		unsigned int						batch_nnodes_min;

		ConditionalOStream 					mcout;

		std::vector<std::string>			qpreplogloc;
		std::vector<std::string>			lengthoutputfile;
		std::vector<std::string>			stressoutputfile;
		std::vector<std::string>			stiffoutputfile;
		std::vector<std::string>			systemoutputfile;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;
		std::vector<ReplicaData<dim> > 		replica_data;
		Tensor<1,dim> 						cg_dir;

		double								md_timestep_length;
		double								md_temperature;
		int									md_nsteps_sample;
		int									md_nsteps_equil;
		double								md_strain_rate;
		double								md_strain_ampl;
		std::string							md_force_field;

		std::string                         nanostatelocin;
		std::string							nanologloc;
		std::string							nanologloctmp;

		std::string							md_scripts_directory;
		bool								use_pjm_scheduler;

	};



	template <int dim>
	EQMDSync<dim>::EQMDSync (MPI_Comm mcomm, int pcolor)
	:
		mmd_communicator (mcomm),
		mmd_n_processes (Utilities::MPI::n_mpi_processes(mmd_communicator)),
		this_mmd_process (Utilities::MPI::this_mpi_process(mmd_communicator)),
		mmd_pcolor (pcolor),
		mcout (std::cout,(this_mmd_process == 0))
	{}



	template <int dim>
	EQMDSync<dim>::~EQMDSync ()
	{}



	// There are several number of processes encountered: (i) n_lammps_processes the highest provided
	// as an argument to aprun, (ii) ND the number of processes provided to deal.ii
	// [arbitrary], (iii) NI the number of processes provided to the lammps initiation
	// [as close as possible to n_world_processes], and (iv) n_lammps_processes_per_batch the number of processes provided to one lammps
	// testing [NT divided by n_lammps_batch the number of concurrent testing boxes].
	template <int dim>
	void EQMDSync<dim>::set_md_procs (int nmdruns)
	{
		// Dispatch of the available processes on to different groups for parallel
		// update of quadrature points
		int npbtch_min = batch_nnodes_min*machine_ppn;
		//int nbtch_max = int(n_world_processes/npbtch_min);

		//int nrounds = int(nmdruns/nbtch_max)+1;
		//int nmdruns_round = nmdruns/nrounds;

		int fair_npbtch = int(mmd_n_processes/(nmdruns));

		int npb = std::max(npbtch_min, fair_npbtch - fair_npbtch%npbtch_min);
		//int nbtch = int(n_world_processes/npbtch);

		// Arbitrary setting of NB and NT
		md_batch_n_processes = npb;

		n_md_batches = int(mmd_n_processes/md_batch_n_processes);
		if(n_md_batches == 0) {n_md_batches=1; md_batch_n_processes=mmd_n_processes;}

		mcout << "        " << "...number of processes per batches: " << md_batch_n_processes
							<< "   ...number of batches: " << n_md_batches << std::endl;

		md_batch_pcolor = MPI_UNDEFINED;

		// LAMMPS processes color: regroup processes by batches of size NB, except
		// the last ones (me >= NB*NC) to create batches of only NB processes, nor smaller.
		if(this_mmd_process < md_batch_n_processes*n_md_batches)
			md_batch_pcolor = int(this_mmd_process/md_batch_n_processes);
		// Initially we used MPI_UNDEFINED, but why waste processes... The processes over
		// 'n_lammps_processes_per_batch*n_lammps_batch' are assigned to the last batch...
		// finally it is better to waste them than failing the simulation with an odd number
		// of processes for the last batch
		/*else
			mmd_pcolor = int((md_batch_n_processes*n_md_batches-1)/md_batch_n_processes);
		*/

		// Definition of the communicators
		MPI_Comm_split(mmd_communicator, md_batch_pcolor, this_mmd_process, &md_batch_communicator);
		MPI_Comm_rank(md_batch_communicator,&this_md_batch_process);

	}




	template <int dim>
	void EQMDSync<dim>::load_replica_generation_data ()
	{
	    using boost::property_tree::ptree;

	    char filename[1024];
		for(unsigned int imd=0; imd<mdtype.size(); imd++)
			for(unsigned int irep=0; irep<nrepl; irep++){
				sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(), mdtype[imd].c_str(), irep+1);
				if(!file_exists(filename)){
					std::cerr << "Missing data for replica #" << irep+1
							  << " of material" << mdtype[imd].c_str()
							  << "." << std::endl;
					exit(1);
				}
			}

		replica_data.resize(nrepl * mdtype.size());
		for(unsigned int imd=0; imd<mdtype.size(); imd++)
			for(unsigned int irep=0; irep<nrepl; irep++){
				// Setting material name and replica number
				replica_data[imd*nrepl+irep].mat=mdtype[imd];
				replica_data[imd*nrepl+irep].repl=irep+1;

				// Initializing mechanical characteristics after equilibration
				replica_data[imd*nrepl+irep].init_length = 0;
				replica_data[imd*nrepl+irep].init_stress = 0;

				// Parse JSON data file
			    sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(),
			    		replica_data[imd*nrepl+irep].mat.c_str(), replica_data[imd*nrepl+irep].repl);

			    std::ifstream jsonFile(filename);
			    ptree pt;
			    try{
				    read_json(jsonFile, pt);
			    }
			    catch (const boost::property_tree::json_parser::json_parser_error& e)
			    {
			        mcout << "Invalid JSON replica data input file (" << filename << ")" << std::endl;  // Never gets here
			    }

			    // Printing the whole tree of the JSON file
			    //bptree_print(pt);

				// Load density of given replica of given material
			    std::string rdensity = bptree_read(pt, "relative_density");
				replica_data[imd*nrepl+irep].rho = std::stod(rdensity)*1000.;

				// Load number of flakes in box
			    std::string numflakes = bptree_read(pt, "Nsheets");
				replica_data[imd*nrepl+irep].nflakes = std::stoi(numflakes);

				/*mcout << "Hi repl: " << replica_data[imd*nrepl+irep].repl
					  << " - mat: " << replica_data[imd*nrepl+irep].mat
					  << " - rho: " << replica_data[imd*nrepl+irep].rho
					  << std::endl;*/

				// Load replica orientation (normal to flake plane if composite)
				if(replica_data[imd*nrepl+irep].nflakes==1){
					std::string fvcoorx = bptree_read(pt, "normal_vector","1","x");
					std::string fvcoory = bptree_read(pt, "normal_vector","1","y");
					std::string fvcoorz = bptree_read(pt, "normal_vector","1","z");
					//hcout << fvcoorx << " " << fvcoory << " " << fvcoorz <<std::endl;
					Tensor<1,dim> nvrep;
					nvrep[0]=std::stod(fvcoorx);
					nvrep[1]=std::stod(fvcoory);
					nvrep[2]=std::stod(fvcoorz);
					// Set the rotation matrix from the replica orientation to common
					// ground FE/MD orientation (arbitrary choose x-direction)
					replica_data[imd*nrepl+irep].rotam=compute_rotation_tensor(nvrep,cg_dir);
				}
				else{
					Tensor<2,dim> idmat;
					idmat = 0.0; for (unsigned int i=0; i<dim; ++i) idmat[i][i] = 1.0;
					// Simply fill the rotation matrix with the identity matrix
					replica_data[imd*nrepl+irep].rotam=idmat;
				}
			}
	}





	template <int dim>
	void EQMDSync<dim>::prepare_replica_equilibration ()
	{
		// Number of MD simulations at this iteration...
		int nmdruns = mdtype.size()*nrepl;

		// Setting up batch of processes
		set_md_procs(nmdruns);

		qpreplogloc.resize(nmdruns,"");
		lengthoutputfile.resize(nmdruns,"");
		stressoutputfile.resize(nmdruns,"");
		stiffoutputfile.resize(nmdruns,"");
		systemoutputfile.resize(nmdruns,"");

		for(unsigned int imdt=0;imdt<mdtype.size();imdt++)
		{
			// type of MD box (so far PE or PNC)
			std::string mdt = mdtype[imdt];

			for(unsigned int repl=0;repl<nrepl;repl++)
			{
				int imdrun=imdt*nrepl + (repl);

				// Offset replica number because in filenames, replicas start at 1
				int numrepl = repl+1;

				// Allocation of a MD run to a batch of processes
				if (md_batch_pcolor == (imdrun%n_md_batches)){
					lengthoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] + "_" + std::to_string(numrepl) + ".length";
					stressoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] + "_" + std::to_string(numrepl) + ".stress";
					stiffoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] + "_" + std::to_string(numrepl) + ".stiff";
					systemoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] + "_" + std::to_string(numrepl) + ".bin";
					qpreplogloc[imdrun] = nanologloctmp + "/init" + "." + mdtype[imdt] + "_" + std::to_string(numrepl);
					if(this_md_batch_process == 0) mkdir(qpreplogloc[imdrun].c_str(), ACCESSPERMS);
				}
			}
		}
	}




	template <int dim>
	void EQMDSync<dim>::equilibrate_replicas ()
	{
		for(unsigned int imdt=0;imdt<mdtype.size();imdt++)
		{
			for(unsigned int repl=0;repl<nrepl;repl++)
			{
				int imdrun=imdt*nrepl + (repl);
				if (md_batch_pcolor == (imdrun%n_md_batches))
				{
					// Offset replica number because in filenames, replicas start at 1
					int numrepl = repl+1;

					// Executing directly from the current MPI_Communicator (not fault tolerant)
					EQMDProblem<3> eqmd_problem (md_batch_communicator, md_batch_pcolor);

					eqmd_problem.equil(mdtype[imdt], nanostatelocin,
								   qpreplogloc[imdrun], md_scripts_directory,
								   lengthoutputfile[imdrun], stressoutputfile[imdrun],
								   stiffoutputfile[imdrun], systemoutputfile[imdrun],
								   numrepl, md_timestep_length, md_temperature,
								   md_nsteps_sample, md_nsteps_equil, md_strain_rate,
								   md_strain_ampl, md_force_field);
				}
			}
		}
	}

	template <int dim>
	void EQMDSync<dim>::equilibrate (double mdtlength, double mdtemp, int nss, int nse, double strr, double stra,
			   std::string ffi, std::string nslocin, std::string nlogloc,
			   std::string nlogloctmp,
			   std::string mdsdir, unsigned int bnmin, unsigned int mppn,
			   std::vector<std::string> mdt, Tensor<1,dim> cgd, unsigned int nr, bool ups){

		md_timestep_length = mdtlength;
		md_temperature = mdtemp;
		md_nsteps_sample = nss;
		md_nsteps_equil = nse;
		md_strain_rate = strr;
		md_strain_ampl = stra;
		md_force_field = ffi;

		nanostatelocin = nslocin;
		nanologloc = nlogloc;
		nanologloctmp = nlogloctmp;

		md_scripts_directory = mdsdir;

		batch_nnodes_min = bnmin;
		machine_ppn = mppn;

		mdtype = mdt;
		cg_dir = cgd;
		nrepl = nr;

		use_pjm_scheduler = ups;

		load_replica_generation_data();
		prepare_replica_equilibration();
		equilibrate_replicas();
	}
}

#endif
