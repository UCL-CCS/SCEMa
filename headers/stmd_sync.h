#ifndef STMD_SYNC_H
#define STMD_SYNC_H

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/foreach.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "math_calc.h"
#include "md_sim.h"
#include "read_write.h"
#include "scale_bridging_data.h"
#include "stmd_problem.h"

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef MIN
#undef MAX

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>

namespace HMM {
using namespace dealii;

template <int dim> struct ReplicaData {
  // Characteristics
  double rho;
  std::string mat;
  int repl;
  int nflakes;
  Tensor<1, dim> init_length;
  Tensor<2, dim> rotam;
  SymmetricTensor<2, dim> init_stress;
  SymmetricTensor<4, dim> init_stiff;
};

template <int dim> class STMDSync {
public:
  STMDSync(MPI_Comm mcomm, int pcolor);
  ~STMDSync();
  void init(int sstp, double mdtlength, double mdtemp, int nss, double strr,
            std::string ffi, std::string nslocin, std::string nslocout,
            std::string nslocres, std::string nlogloc, std::string mslocout,
            std::string mdsdir, int fchpt, int fohom, unsigned int mppn,
            std::vector<std::string> mdt, Tensor<1, dim> cgd, unsigned int nr,
            bool ups, boost::property_tree::ptree inconfig,
            bool approx_md_with_hookes_law);
  void update(int tstp, double ime, int nstp,
              ScaleBridgingData &scale_bridging_data);

private:
  void restart();

  void set_md_procs(int nmdruns);

  void load_replica_generation_data();
  void load_replica_equilibration_data();

  void average_replica_data();

  std::vector<MDSim<dim>>
  prepare_md_simulations(ScaleBridgingData scale_bridging_data);

  void
  execute_inside_md_simulations(std::vector<MDSim<dim>> &requested_simulations);
  void share_stresses(std::vector<MDSim<dim>> &md_simulations);

  void write_exec_script_md_job();
  void generate_job_list(bool &elmj, int &tta, char *filenamelist);
  void execute_pjm_md_simulations();

  void store_md_simulations(std::vector<MDSim<dim>> md_simulations,
                            ScaleBridgingData &scale_bridging_data);
  void p_store_md_simulations(std::vector<MDSim<dim>> md_simulations,
                              ScaleBridgingData &scale_bridging_data);

  MPI_Comm mmd_communicator;
  MPI_Comm md_batch_communicator;
  uint32_t mmd_n_processes;
  uint32_t md_batch_n_processes;
  uint32_t n_md_batches;
  int32_t this_mmd_process;
  int32_t this_md_batch_process;
  int32_t mmd_pcolor;
  int32_t md_batch_pcolor;

  uint32_t ncupd;

  uint32_t machine_ppn;

  ConditionalOStream mcout;

  int start_timestep;
  double present_time;
  int timestep;
  int newtonstep;

  std::string time_id;
  std::vector<std::string> cell_id;
  std::vector<std::string> cell_mat;

  std::vector<std::string> qpreplogloc;
  std::vector<std::string> straininputfile;
  std::vector<std::string> lengthoutputfile;
  std::vector<std::string> stressoutputfile;
  std::vector<std::string> stiffoutputfile;
  std::vector<std::string> systemoutputfile;

  std::vector<std::string> mdtype;
  uint32_t nrepl;
  std::vector<ReplicaData<dim>> replica_data;
  Tensor<1, dim> cg_dir;

  double md_timestep_length;
  double md_temperature;
  int md_nsteps_sample;
  double md_strain_rate;
  std::string md_force_field;

  std::vector<std::vector<std::string>> md_args;

  int freq_checkpoint;
  int freq_output_homog;

  bool output_homog;
  bool checkpoint_save;

  std::string macrostatelocout;

  std::string nanostatelocin;
  std::string nanostatelocout;
  std::string nanostatelocres;
  std::string nanologloc;

  std::string md_scripts_directory;
  bool use_pjm_scheduler;
  boost::property_tree::ptree input_config;

  bool approx_md_with_hookes_law;
};

template <int dim>
STMDSync<dim>::STMDSync(MPI_Comm mcomm, int pcolor)
    : mmd_communicator(mcomm),
      mmd_n_processes(Utilities::MPI::n_mpi_processes(mmd_communicator)),
      this_mmd_process(Utilities::MPI::this_mpi_process(mmd_communicator)),
      mmd_pcolor(pcolor), mcout(std::cout, (this_mmd_process == 0)) {}

template <int dim> STMDSync<dim>::~STMDSync() {}

template <int dim> void STMDSync<dim>::restart() {
  // Could distribute that command over available processes
  // Cleaning the log files for all the MD simulations of the current timestep
  if (this_mmd_process == 0) {
    char command[1024];
    // Clean "nanoscale_logs" of the finished timestep
    sprintf(command,
            "for ii in `ls %s/restart/ | grep -o '[^-]*$' | cut -d. -f2-`; "
            "do cp %s/restart/lcts.${ii} %s/last.${ii}; "
            "done",
            nanostatelocin.c_str(), nanostatelocin.c_str(),
            nanostatelocout.c_str());
    int ret = system(command);
    if (ret != 0) {
      std::cerr << "Failed to copy input restart files (lcts) of the MD "
                   "simulations as current output (last)!"
                << std::endl;
      exit(1);
    }
  }
}

template <int dim> void STMDSync<dim>::set_md_procs(int nmdruns) {
  // Dispatch of the available processes on to different groups for parallel
  // update of quadrature points

  // Setting the minimum possible allocation per MD sim
  unsigned int npbtch_min = input_config.get<unsigned int>(
      "computational resources.minimum number of cores for MD simulation");

  // Setting the number of cores per node
  unsigned int npnode = input_config.get<unsigned int>(
      "computational resources.machine cores per node");

  unsigned int fair_npbtch;
  if (nmdruns > 0) {
    fair_npbtch = int(mmd_n_processes / (nmdruns));
    if (fair_npbtch == 0)
      fair_npbtch = 1;
  } else {
    fair_npbtch = mmd_n_processes;
  }

  // Building the list of admissible core count:
  // multiple or factors of the core count per node
  // in the range from minimum possible allocation (npbtch_min) to total core
  // count (mmd_n_processes)
  std::vector<unsigned int> list_possible_cores_per_job;
  for (unsigned int ic = npbtch_min; ic <= mmd_n_processes; ic++) {
    if (ic <= npnode) {
      if (npnode % ic == 0)
        list_possible_cores_per_job.push_back(ic);
    } else {
      if (ic % npnode == 0)
        list_possible_cores_per_job.push_back(ic);
    }
  }

  // Setting core allocation as the largest admissible core count inferior to
  // the fair core count
  for (unsigned int ic = 0; ic < list_possible_cores_per_job.size(); ic++) {
    if (list_possible_cores_per_job[ic] > fair_npbtch)
      break;
    // Setting the number of processes per md simulation "md_batch_n_processes"
    md_batch_n_processes = list_possible_cores_per_job[ic];
  }

  // Throw error if md_batch_n_processes is not set in the range
  // from minimum possible allocation (npbtch_min) to total core count
  // (mmd_n_processes) and not set as either a factor or a multiple of the
  // number of cores per node (npnode)
  if (md_batch_n_processes < npbtch_min ||
      md_batch_n_processes > mmd_n_processes ||
      (npnode % md_batch_n_processes != 0 and
       md_batch_n_processes % npnode != 0)) {
    std::cout << "Error: The number of cores per MD simulation batch "
                 "(md_batch_n_processes) is not well set!"
              << std::endl;
    exit(1);
  }

  n_md_batches = int(mmd_n_processes / md_batch_n_processes);
  if (n_md_batches == 0) {
    n_md_batches = 1;
    md_batch_n_processes = mmd_n_processes;
  }

  mcout << "        "
        << "...number of processes per batches: " << md_batch_n_processes
        << "   ...number of batches: " << n_md_batches << std::endl;

  md_batch_pcolor = MPI_UNDEFINED;

  // LAMMPS processes color: regroup processes by batches of size NB, except
  // the last ones (me >= NB*NC) to create batches of only NB processes, nor
  // smaller.
  if (this_mmd_process < md_batch_n_processes * n_md_batches)
    md_batch_pcolor = int(this_mmd_process / md_batch_n_processes);
  // Initially we used MPI_UNDEFINED, but why waste processes... The processes
  // over 'n_lammps_processes_per_batch*n_lammps_batch' are assigned to the last
  // batch... finally it is better to waste them than failing the simulation
  // with an odd number of processes for the last batch
  /*else
                  mmd_pcolor =
     int((md_batch_n_processes*n_md_batches-1)/md_batch_n_processes);
   */

  // Definition of the communicators
  MPI_Comm_split(mmd_communicator, md_batch_pcolor, this_mmd_process,
                 &md_batch_communicator);
  MPI_Comm_rank(md_batch_communicator, &this_md_batch_process);
}

template <int dim> void STMDSync<dim>::load_replica_generation_data() {
  using boost::property_tree::ptree;

  char filename[1024];
  for (unsigned int imd = 0; imd < mdtype.size(); imd++)
    for (unsigned int irep = 0; irep < nrepl; irep++) {
      sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(),
              mdtype[imd].c_str(), irep + 1);
      if (!file_exists(filename)) {
        std::cerr << "Missing data for replica #" << irep + 1 << " of material"
                  << mdtype[imd].c_str() << "." << std::endl;
        exit(1);
      }
    }

  replica_data.resize(nrepl * mdtype.size());
  for (unsigned int imd = 0; imd < mdtype.size(); imd++)
    for (unsigned int irep = 0; irep < nrepl; irep++) {
      // Setting material name and replica number
      replica_data[imd * nrepl + irep].mat = mdtype[imd];
      replica_data[imd * nrepl + irep].repl = irep + 1;

      // Initializing mechanical characteristics after equilibration
      replica_data[imd * nrepl + irep].init_length = 0;
      replica_data[imd * nrepl + irep].init_stress = 0;

      // Parse JSON data file
      sprintf(filename, "%s/%s_%d.json", nanostatelocin.c_str(),
              replica_data[imd * nrepl + irep].mat.c_str(),
              replica_data[imd * nrepl + irep].repl);

      std::ifstream jsonFile(filename);
      ptree pt;
      try {
        read_json(jsonFile, pt);
      } catch (const boost::property_tree::json_parser::json_parser_error &e) {
        mcout << "Invalid JSON replica data input file (" << filename << ")"
              << std::endl; // Never gets here
      }

      // Printing the whole tree of the JSON file
      // bptree_print(pt);

      // Load density of given replica of given material
      std::string rdensity = bptree_read(pt, "relative_density");
      replica_data[imd * nrepl + irep].rho = std::stod(rdensity) * 1000.;

      // Load number of flakes in box
      std::string numflakes = bptree_read(pt, "Nsheets");
      replica_data[imd * nrepl + irep].nflakes = std::stoi(numflakes);

      /*mcout << "Hi repl: " << replica_data[imd*nrepl+irep].repl
                        << " - mat: " << replica_data[imd*nrepl+irep].mat
                        << " - rho: " << replica_data[imd*nrepl+irep].rho
                        << std::endl;*/

      // Load replica orientation (normal to flake plane if composite)
      if (replica_data[imd * nrepl + irep].nflakes == 1) {
        std::string fvcoorx = bptree_read(pt, "normal_vector", "1", "x");
        std::string fvcoory = bptree_read(pt, "normal_vector", "1", "y");
        std::string fvcoorz = bptree_read(pt, "normal_vector", "1", "z");
        // hcout << fvcoorx << " " << fvcoory << " " << fvcoorz <<std::endl;
        Tensor<1, dim> nvrep;
        nvrep[0] = std::stod(fvcoorx);
        nvrep[1] = std::stod(fvcoory);
        nvrep[2] = std::stod(fvcoorz);
        // Set the rotation matrix from the replica orientation to common
        // ground FE/MD orientation (arbitrary choose x-direction)
        replica_data[imd * nrepl + irep].rotam =
            compute_rotation_tensor(nvrep, cg_dir);
      } else {
        Tensor<2, dim> idmat;
        idmat = 0.0;
        for (unsigned int i = 0; i < dim; ++i)
          idmat[i][i] = 1.0;
        // Simply fill the rotation matrix with the identity matrix
        replica_data[imd * nrepl + irep].rotam = idmat;
      }
    }
}

template <int dim> void STMDSync<dim>::load_replica_equilibration_data() {

  // Number of MD simulations at this iteration...
  int nmdruns = mdtype.size() * nrepl;

  qpreplogloc.resize(nmdruns, "");
  lengthoutputfile.resize(nmdruns, "");
  stressoutputfile.resize(nmdruns, "");
  stiffoutputfile.resize(nmdruns, "");
  systemoutputfile.resize(nmdruns, "");

  for (unsigned int imdt = 0; imdt < mdtype.size(); imdt++) {
    // type of MD box (so far PE or PNC)
    std::string mdt = mdtype[imdt];

    for (unsigned int repl = 0; repl < nrepl; repl++) {
      int imdrun = imdt * nrepl + (repl);

      // Offset replica number because in filenames, replicas start at 1
      int numrepl = repl + 1;

      lengthoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] +
                                 "_" + std::to_string(numrepl) + ".length";
      stressoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] +
                                 "_" + std::to_string(numrepl) + ".stress";
      stiffoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] + "_" +
                                std::to_string(numrepl) + ".stiff";
      systemoutputfile[imdrun] = nanostatelocin + "/init." + mdtype[imdt] +
                                 "_" + std::to_string(numrepl) + ".bin";
      qpreplogloc[imdrun] = nanologloc + "/init" + "." + mdtype[imdt] + "_" +
                            std::to_string(numrepl);
    }
  }

  for (unsigned int imd = 0; imd < mdtype.size(); imd++)
    for (unsigned int irep = 0; irep < nrepl; irep++) {

      int imdrun = imd * nrepl + (irep);

      // Load replica initial dimensions
      bool statelength_exists = file_exists(lengthoutputfile[imdrun].c_str());
      if (statelength_exists) {
        read_tensor<dim>(lengthoutputfile[imdrun].c_str(),
                         replica_data[imdrun].init_length);
      } else {
        std::cerr << "Missing equilibrated initial length data for material "
                  << replica_data[imdrun].mat.c_str() << " replica #"
                  << replica_data[imdrun].repl << std::endl;
      }

      // Load replica initial stresses
      bool statestress_exists = file_exists(stressoutputfile[imdrun].c_str());
      if (statestress_exists) {
        read_tensor<dim>(stressoutputfile[imdrun].c_str(),
                         replica_data[imdrun].init_stress);
      } else {
        std::cerr << "Missing equilibrated initial stress data for material "
                  << replica_data[imdrun].mat.c_str() << " replica #"
                  << replica_data[imdrun].repl << std::endl;
      }

      // Load replica initial stiffness
      bool statestiff_exists = file_exists(stiffoutputfile[imdrun].c_str());
      if (statestiff_exists) {
        read_tensor<dim>(stiffoutputfile[imdrun].c_str(),
                         replica_data[imdrun].init_stiff);
      } else {
        std::cerr << "Missing equilibrated initial stiffness data for material "
                  << replica_data[imdrun].mat.c_str() << " replica #"
                  << replica_data[imdrun].repl << std::endl;
      }

      if (this_mmd_process == 0) {
        // Copying replica input system
        bool statesystem_exists = file_exists(systemoutputfile[imdrun].c_str());
        if (statesystem_exists) {
          std::ifstream nanoin(systemoutputfile[imdrun].c_str(),
                               std::ios::binary);
          char nanofilenameout[1024];
          sprintf(nanofilenameout, "%s/init.%s_%d.bin", nanostatelocout.c_str(),
                  replica_data[imdrun].mat.c_str(), replica_data[imdrun].repl);
          std::ofstream nanoout(nanofilenameout, std::ios::binary);
          nanoout << nanoin.rdbuf();
          nanoin.close();
          nanoout.close();
        } else {
          std::cerr << "Missing equilibrated initial system for material "
                    << replica_data[imdrun].mat.c_str() << " replica #"
                    << replica_data[imdrun].repl << std::endl;
        }
      }
    }
}

template <int dim> void STMDSync<dim>::average_replica_data() {
  for (unsigned int imd = 0; imd < mdtype.size(); imd++) {
    SymmetricTensor<4, dim> initial_stiffness_tensor;
    initial_stiffness_tensor = 0.;

    double initial_density = 0.;

    for (unsigned int repl = 0; repl < nrepl; repl++) {
      SymmetricTensor<4, dim> cg_initial_rep_stiffness_tensor;

      // Rotate tensor from replica orientation to common ground
      cg_initial_rep_stiffness_tensor =
          rotate_tensor(replica_data[imd * nrepl + repl].init_stiff,
                        replica_data[imd * nrepl + repl].rotam);

      // Averaging tensors in the common ground referential
      initial_stiffness_tensor += cg_initial_rep_stiffness_tensor;

      // Averaging density over replicas
      initial_density += replica_data[imd * nrepl + repl].rho;
    }

    initial_stiffness_tensor /= nrepl;
    initial_density /= nrepl;

    char macrofilenameout[1024];
    sprintf(macrofilenameout, "%s/init.%s.stiff", macrostatelocout.c_str(),
            mdtype[imd].c_str());
    write_tensor<dim>(macrofilenameout, initial_stiffness_tensor);

    sprintf(macrofilenameout, "%s/init.%s.density", macrostatelocout.c_str(),
            mdtype[imd].c_str());
    write_tensor<dim>(macrofilenameout, initial_density);
  }
}

template <int dim>
std::vector<MDSim<dim>>
STMDSync<dim>::prepare_md_simulations(ScaleBridgingData scale_bridging_data) {
  std::vector<MDSim<dim>> request_simulations;
  std::vector<QP> update_list = scale_bridging_data.update_list;

  uint32_t n_qp = update_list.size();

  // Number of MD simulations at this iteration...
  uint32_t nmdruns = n_qp * nrepl;

  // Setting up batch of processes
  set_md_procs(nmdruns);

  for (uint32_t qp = 0; qp < n_qp; ++qp) {
    for (uint32_t repl = 0; repl < nrepl; repl++) {
      // Allocation of a MD run to a batch of processes
      // if (md_batch_pcolor == (imdrun%n_md_batches)){

      MDSim<dim> md_sim;
      md_sim.qp_id = update_list[qp].id;
      md_sim.most_recent_qp_id = update_list[qp].most_recent_id;

      md_sim.replica = repl + 1; // +1 to match input file lables... fix
      md_sim.material = update_list[qp].material;

      int replica_data_index =
          md_sim.material * nrepl + repl; // imd*nrepl+nrepl;
      md_sim.matid = replica_data[replica_data_index].mat;
      md_sim.time_id = time_id;

      md_sim.force_field = md_force_field;
      md_sim.timestep_length = md_timestep_length;
      md_sim.temperature = md_temperature;
      md_sim.nsteps_sample = md_nsteps_sample;
      md_sim.strain_rate = md_strain_rate;

      md_sim.output_folder = nanostatelocout;
      md_sim.restart_folder = nanostatelocres;
      md_sim.scripts_folder = md_scripts_directory;
      md_sim.output_homog = false;
      md_sim.checkpoint = checkpoint_save;
      // Setting up location for temporary log outputs of md simulation, input
      // strains and output stresses
      std::string macrostatelocout = input_config.get<std::string>(
          "directory structure.macroscale output");
      if (approx_md_with_hookes_law == false)
        md_sim.define_file_names(nanologloc);

      // Argument of the MD simulation: strain to apply
      SymmetricTensor<2, dim> cg_loc_rep_strain(
          scale_bridging_data.update_list[qp].update_strain);

      // Rotate strain tensor from common ground to replica orientation
      md_sim.strain = rotate_tensor(
          cg_loc_rep_strain, transpose(replica_data[replica_data_index].rotam));
      // Resize applied strain with initial length of the md sample, the
      // resulting variable is not a strain but a length variation, which will
      // be transformed back into a strain during the execution of the MD code
      // where the current length of the nanosystem will be available
      if (approx_md_with_hookes_law == false) {
        for (unsigned int j = 0; j < dim; j++) {
          md_sim.strain[j][j] *=
              replica_data[replica_data_index].init_length[j];
          md_sim.strain[j][(j + 1) % dim] *=
              replica_data[replica_data_index].init_length[(j + 2) % dim];
        }
      }

      // Setting up md system stiffness tensor
      md_sim.stiffness = replica_data[replica_data_index].init_stiff;

      request_simulations.push_back(md_sim);
    }
  }

  return request_simulations;
}

template <int dim>
void STMDSync<dim>::execute_inside_md_simulations(
    std::vector<MDSim<dim>> &md_simulations) {
  // Computing cell state update running one simulation per MD replica (basic
  // job scheduling and executing)
  mcout << "        "
        << "...dispatching the MD runs on batch of processes..." << std::endl;
  mcout << "        "
        << "...cells and replicas completed: " << std::flush;
  uint32_t n_md_runs = md_simulations.size();

  for (uint32_t i = 0; i < n_md_runs; ++i) {
    // Allocation of a MD run to a batch of processes
    if (md_batch_pcolor == (i % n_md_batches)) {
      // Executing from an external MPI_Communicator (avoids failure of the main
      // communicator when the specific/external communicator fails) Does not
      // work as OpenMPI cannot be started from an existing OpenMPI run...
      /*std::string exec_name = "mpirun ./single_md";

                      // Writting the argument list to be passed to the
         strain_md executable directly std::vector<std::string>
         args_list_separator = " "; std::string args_list; for (int i=0;
         i<md_args[imdrun].size(); i++){ args_list +=
         args_list_separator+md_args[i];
                      }

                      std::string redir_output = " > " + qpreplogloc[imdrun] +
         "/out.single_md";

                      std::string command = exec_name+args_list+redir_output;
                      int ret = system(command.c_str());
                      if (ret!=0){
                              std::cerr << "Failed executing the md simulation:
         " << command << std::endl; exit(1);
                      }*/

      // Executing directly from the current MPI_Communicator (not fault
      // tolerant)

      // MPI_Barrier(mmd_communicator);
      STMDProblem<3> stmd_problem(md_batch_communicator, md_batch_pcolor);

      stmd_problem.strain(md_simulations[i], approx_md_with_hookes_law);
    }
  }
  mcout << std::endl;

  MPI_Barrier(mmd_communicator);
}

template <int dim>
void STMDSync<dim>::share_stresses(std::vector<MDSim<dim>> &md_simulations) {
  // share all the stresses calculated in different batches to rank 0

  MPI_Request request;
  MPI_Status status;

  uint32_t n_md_runs = md_simulations.size();

  std::vector<int> pcolors(mmd_n_processes);
  for (unsigned int i = 0; i < mmd_n_processes; i++) {
    pcolors[i] =
        int(i / md_batch_n_processes); // this is the commmand in set_md_procs
  }

  if (pcolors[0] != 0) {
    std::cout << "Error: root rank has pcolor != 0" << std::endl;
    exit(1);
  }

  // for eah batch, send stresses it calculated to rank 0
  for (int32_t pcolor = 1; pcolor < n_md_batches; pcolor++) {

    // find lowest rank in this batch - this will be used to send the batch's
    // information
    int32_t batch_root;
    bool found = false;
    for (uint32_t i = 0; i < mmd_n_processes; i++) {
      if (pcolors[i] == pcolor) {
        batch_root = i;
        found = true;
        break;
      }
    }
    assert(found == true);

    // number of sims to share from this batch
    uint32_t n_sends = 0;
    if (this_mmd_process == batch_root) {
      for (uint32_t i = 0; i < n_md_runs; i++) {
        if (md_simulations[i].stress_updated == true) {
          n_sends++;
        }
      }
    }
    MPI_Bcast(&n_sends, 1, MPI_INT, batch_root, mmd_communicator);

    // ids of sims to share from this batch
    // int md_send_indexes[n_sends];
    std::vector<uint32_t> md_send_indexes(n_sends);
    uint32_t count = 0;
    if (this_mmd_process == batch_root) {
      for (uint32_t i = 0; i < n_md_runs; i++) {
        if (md_simulations[i].stress_updated == true) {
          md_send_indexes[count] = i;
          count++;
        }
      }
    }
    MPI_Bcast(&md_send_indexes[0], n_sends, MPI_INT, batch_root,
              mmd_communicator);

    // for each md run calculated on this batch
    for (uint32_t i = 0; i < n_sends; i++) {
      uint32_t md_run_index = md_send_indexes[i];

      // share stresses calculated to rank 0
      if (this_mmd_process == batch_root) {
        // build array to send
        double send_stress[6];
        for (uint32_t j = 0; j < 6; j++) {
          send_stress[j] =
              md_simulations[md_run_index].stress.access_raw_entry(j);
        }
        // send to rank 0
        MPI_Isend(&send_stress[0], 6, MPI_DOUBLE, 0, md_run_index,
                  mmd_communicator, &request);

      } else if (this_mmd_process == 0) {
        // recieve stress and convert to symmetric tensor
        double recv_stress[6];
        MPI_Recv(&recv_stress[0], 6, MPI_DOUBLE, batch_root, md_run_index,
                 mmd_communicator, &status);
        SymmetricTensor<2, dim> stress(recv_stress);

        md_simulations[md_run_index].stress = stress;
        md_simulations[md_run_index].stress_updated = true;
      }
    }
  }

  // Check all stresses are set on rank 0
  if (this_mmd_process == 0) {
    for (uint32_t i = 0; i < md_simulations.size(); i++) {
      if (md_simulations[i].stress_updated != true) {
        for (uint32_t j = 0; j < 6; j++) {
          std::cout << md_simulations[i].stress.access_raw_entry(j) << " ";
        }
        std::cout << std::endl;
        std::cout << "Stress not set or not communicated to rank ("
                  << this_mmd_process << ") . " << md_simulations[i].qp_id
                  << std::endl;
        exit(1);
      }
    }
  }
}

template <int dim> void STMDSync<dim>::write_exec_script_md_job() {
  for (unsigned int c = 0; c < ncupd; ++c) {
    for (unsigned int repl = 0; repl < nrepl; repl++) {
      // Offset replica number because in filenames, replicas start at 1
      int numrepl = repl + 1;

      // The variable 'imdrun' assigned to a run is a multiple of the batch
      // number the run will be run on
      int imdrun = c * nrepl + (repl);

      if (md_batch_pcolor == (imdrun % n_md_batches)) {
        if (this_md_batch_process == 0) {

          // Writting the argument list to be passed to the JSON file
          std::string args_list_separator = " ";
          std::string args_list = "./strain_md";
          for (unsigned int i = 0; i < md_args[imdrun].size(); i++) {
            args_list += args_list_separator + md_args[imdrun][i];
          }
          args_list += "";

          std::string scriptfile = nanostatelocout + "/" + "bash_cell" +
                                   cell_id[c] + "_repl" +
                                   std::to_string(numrepl) + ".sh";
          std::ofstream bash_script(scriptfile, std::ios_base::trunc);
          bash_script << "mpirun " << args_list;
          bash_script.close();
        }
      }
    }
  }
}

template <int dim>
void STMDSync<dim>::generate_job_list(bool &elmj, int &tta,
                                      char *filenamelist) {
  char command[1024];
  int ret, rval;

  sprintf(command,
          "python ../optimization_pjm/optimization_hmm.py %s %d %d %s %s %s %s",
          macrostatelocout.c_str(), 1, nrepl, time_id.c_str(),
          nanostatelocout.c_str(), nanologloc.c_str(), filenamelist);

  // Executing the job list optimization script with fscanf to parse the printed
  // values from the python script
  FILE *in = popen(command, "r");
  ret = fscanf(in, "%d", &rval);
  pclose(in);
  if (ret != 1) {
    std::cerr << "Failed executing the job list optimization script, incorrect "
                 "number/format of returned values ("
              << ret << ")" << std::endl;
    exit(1);
  }

  // Retrieving output value of the optimization script
  if (rval == 0) {
    elmj = true;
  } else if (rval > 0) {
    tta = rval;
  } else {
    std::cerr << "Invalid returned value (" << rval
              << ") from the job list optimization script." << std::endl;
    exit(1);
  }

  // Verify function returned values
  if (elmj == false && tta == 0) {
    std::cerr << "Invalid combinaison of empty_list_md_jobs and "
                 "total_node_allocation."
              << std::endl;
    exit(1);
  }
}

template <int dim> void STMDSync<dim>::execute_pjm_md_simulations() {
  int total_node_allocation = 0;
  int ret;
  write_exec_script_md_job();

  char filenamelist[1024], command[1024];
  sprintf(filenamelist, "%s/list_md_jobs.json", nanostatelocout.c_str());

  if (this_mmd_process == 0) {

    bool empty_list_md_jobs = false;
    std::cout << "        "
              << "...building optimized job list for pilotjob execution..."
              << std::endl;

    generate_job_list(empty_list_md_jobs, total_node_allocation, filenamelist);

    if (empty_list_md_jobs) {
      std::cout << "          The .json file is empty, no execution of QCG-PM"
                << std::endl;
    } else {

      // Run python script that runs all the MD jobs located in json file
      std::cout << "        "
                << "...calling QCG-PM..." << std::endl;
      // This command will ask for its specific allocation outside of the
      // present one
      sprintf(command,
              "sbatch -p standard -Q -W -A compatpsnc2 -N %d --ntasks-per-node "
              "28 -t 02:00:00 "
              "--wrap='/opt/exp_soft/plgrid/qcg-appscripts-eagle/tools/"
              "qcg-pilotmanager/qcg-pm-service "
              "--exschema slurm --file --file-path=%s'",
              total_node_allocation, filenamelist);
      // This one will be run inside the present allocation
      /*sprintf(command,
                              "/opt/exp_soft/plgrid/qcg-appscripts-eagle/tools/qcg-pilotmanager/qcg-pm-service
         "
                              "--exschema slurm --file --file-path=%s",
                              filenamelist);*/
      ret = system(command);
      if (ret != 0) {
        std::cout << "Failed completing the MD updates via QCG-PM" << std::endl;
        // std::cerr << "Failed completing the MD updates via QCG-PM" <<
        // std::endl; exit(1);
      }

      std::cout << "        "
                << "...completion signal from QCG-PM received!" << std::endl;
    }
  }
}

template <int dim>
int get_sim_id(std::vector<MDSim<dim>> md_simulations, int qp_id, int rep) {
  // find md_simulation for qp_id and rep
  int n_md_sim = md_simulations.size();
  int md_index;
  bool found = false;
  for (int i = 0; i < n_md_sim; i++) {
    if (md_simulations[i].qp_id == qp_id && md_simulations[i].replica == rep) {
      md_index = i;
      found = true;
      break;
    }
  }
  if (found != true) {
    std::cout << "Error: MDSim not found for qp " << qp_id << " replica " << rep
              << std::endl;
    exit(1);
  }
  return md_index;
}

template <int dim>
void STMDSync<dim>::store_md_simulations(
    std::vector<MDSim<dim>> md_simulations,
    ScaleBridgingData &scale_bridging_data) {

  const uint32_t n_qp = scale_bridging_data.update_list.size();

  // Averaging stiffness and stress per cell over replicas
  for (uint32_t qp = 0; qp < n_qp; ++qp) {
    const uint32_t qp_id = scale_bridging_data.update_list[qp].id;

    SymmetricTensor<2, dim> cg_loc_stress;

    for (uint32_t rep = 0; rep < nrepl; ++rep) {
      const uint32_t numrepl = rep + 1;
      const uint32_t md_sim_id = get_sim_id(md_simulations, qp_id, numrepl);

      const MDSim<dim> md_simulation = md_simulations[md_sim_id];

      const uint32_t replica_data_index = md_simulation.material * nrepl + rep;

      // stress from most recent md simulation
      SymmetricTensor<2, dim> loc_rep_stress = md_simulation.stress;

      // subtract the intial stress in the starting structure
      if (approx_md_with_hookes_law == false) {
        loc_rep_stress -= replica_data[replica_data_index].init_stress;
      }

      // Rotation of the stress tensor to common ground direction before
      // averaging
      const SymmetricTensor<2, dim> cg_loc_rep_stress =
          rotate_tensor(loc_rep_stress, replica_data[replica_data_index].rotam);

      cg_loc_stress += cg_loc_rep_stress;
    }
    cg_loc_stress /= nrepl;

    // serialse qp stress into scale bridging data array
    for (uint32_t i = 0; i < 6; ++i) {
      scale_bridging_data.update_list[qp].update_stress[i] =
          cg_loc_stress.access_raw_entry(i);
    }
  }
}

template <int dim>
void STMDSync<dim>::p_store_md_simulations(
    std::vector<MDSim<dim>> md_simulations,
    ScaleBridgingData &scale_bridging_data) {

  const uint32_t n_qp = scale_bridging_data.update_list.size();
  const uint32_t n_processes_per_qp = (float)mmd_n_processes / (float)n_qp;

  if (this_mmd_process == 0) {

    for (uint32_t qp = 0; qp < n_qp; ++qp) {

      const uint32_t qp_id = scale_bridging_data.update_list[qp].id;

      SymmetricTensor<2, dim> cg_loc_stress;

      for (uint32_t rep = 0; rep < nrepl; ++rep) {

        double cg_loc_rep_stress_arr[6]{0};

        const uint32_t numrepl = rep + 1;
        const uint32_t md_sim_id = get_sim_id(md_simulations, qp_id, numrepl);

        const MDSim<dim> md_simulation = md_simulations[md_sim_id];

        const uint32_t replica_data_index =
            md_simulation.material * nrepl + rep;

        // stress from most recent md simulation
        SymmetricTensor<2, dim> loc_rep_stress = md_simulation.stress;

        // subtract the intial stress in the starting structure
        if (approx_md_with_hookes_law == false) {
          loc_rep_stress -= replica_data[replica_data_index].init_stress;
        }

        // Rotation of the stress tensor to common ground direction before
        // averaging
        const SymmetricTensor<2, dim> cg_loc_rep_stress = rotate_tensor(
            loc_rep_stress, replica_data[replica_data_index].rotam);

        for (uint32_t i = 0; i < 6; i++) {
          cg_loc_rep_stress_arr[i] = cg_loc_rep_stress.access_raw_entry(i);
        }

        MPI_Send(cg_loc_rep_stress_arr, 6, MPI_DOUBLE, qp + 1, qp,
                 mmd_communicator);
      }
    }

    for (uint32_t qp = 0; qp < n_qp; ++qp) {

      MPI_Status status;
      double cg_loc_stress_arr[6]{0};

      MPI_Recv(cg_loc_stress_arr, 6, MPI_DOUBLE, qp + 1, 0, mmd_communicator,
               &status);

      for (uint32_t i = 0; i < 6; ++i) {
        scale_bridging_data.update_list[qp].update_stress[i] =
            cg_loc_stress_arr[i];
      }
    }

  } else {

    const uint32_t qp = this_mmd_process - 1;
    const uint32_t qp_id = scale_bridging_data.update_list[qp].id;

    SymmetricTensor<2, dim> cg_loc_stress;

    for (uint32_t rep = 0; rep < nrepl; ++rep) {

      MPI_Status status;
      double cg_loc_rep_stress_arr[6]{0};
      SymmetricTensor<2, dim> cg_loc_rep_stress;

      MPI_Recv(cg_loc_rep_stress_arr, 6, MPI_DOUBLE, 0, qp, mmd_communicator,
               &status);

      for (uint32_t i = 0; i < 6; ++i) {
        cg_loc_rep_stress.access_raw_entry(i) = cg_loc_rep_stress_arr[i];
      }

      cg_loc_stress += cg_loc_rep_stress;
    }

    cg_loc_stress /= nrepl;

    double cg_loc_stress_arr[6]{0};

    for (uint32_t i = 0; i < 6; ++i) {
      cg_loc_stress_arr[i] = cg_loc_stress.access_raw_entry(i);
    }

    MPI_Send(cg_loc_stress_arr, 6, MPI_DOUBLE, 0, 0, mmd_communicator);
  }
}

template <int dim>
void STMDSync<dim>::init(int sstp, double mdtlength, double mdtemp, int nss,
                         double strr, std::string ffi, std::string nslocin,
                         std::string nslocout, std::string nslocres,
                         std::string nlogloc, std::string mslocout,
                         std::string mdsdir, int fchpt, int fohom,
                         unsigned int mppn, std::vector<std::string> mdt,
                         Tensor<1, dim> cgd, unsigned int nr, bool ups,
                         boost::property_tree::ptree inconfig, bool hookeslaw) {

  approx_md_with_hookes_law = hookeslaw;

  input_config = inconfig;
  start_timestep = sstp;

  md_timestep_length = mdtlength;
  md_temperature = mdtemp;
  md_nsteps_sample = nss;
  md_strain_rate = strr;
  md_force_field = ffi;

  nanostatelocin = nslocin;
  nanostatelocout = nslocout;
  nanostatelocres = nslocres;
  nanologloc = nlogloc;

  macrostatelocout = mslocout;
  md_scripts_directory = mdsdir;

  freq_checkpoint = fchpt;
  freq_output_homog = fohom;

  machine_ppn = mppn;

  mdtype = mdt;
  cg_dir = cgd;
  nrepl = nr;

  use_pjm_scheduler = ups;
  restart();
  load_replica_generation_data();
  load_replica_equilibration_data();
  if (this_mmd_process == 0) {
    average_replica_data();
  }
}

template <int dim>
void STMDSync<dim>::update(int tstp, double ptime, int nstp,
                           ScaleBridgingData &scale_bridging_data) {
  present_time = ptime;
  timestep = tstp;
  newtonstep = nstp;

  cell_id.clear();
  cell_mat.clear();
  time_id = std::to_string(timestep) + "-" + std::to_string(newtonstep);
  qpreplogloc.clear();
  md_args.clear();

  // Should the homogenization trajectory file be saved?
  if (timestep % freq_output_homog == 0)
    output_homog = true;
  else
    output_homog = false;

  if (timestep % freq_checkpoint == 0)
    checkpoint_save = true;
  else
    checkpoint_save = false;

  std::vector<MDSim<dim>> md_simulations;
  md_simulations = prepare_md_simulations(scale_bridging_data);

  MPI_Barrier(mmd_communicator);
  int n_md = md_simulations.size();
  mcout << "        Running " << n_md << " simulations:\n";
  for (int i = 0; i < n_md; i++) {
    mcout << md_simulations[i].qp_id << "-" << md_simulations[i].replica << " ";
  }
  mcout << std::endl;

  if (n_md > 0) {
    if (use_pjm_scheduler) {
      execute_pjm_md_simulations();
    } else {
      execute_inside_md_simulations(md_simulations);

      MPI_Barrier(mmd_communicator);

      share_stresses(md_simulations);
    }

    const uint32_t n_qp = scale_bridging_data.update_list.size();

    // average stresses over md replicas, and store them in scale_bridging_data
    if (n_qp > mmd_n_processes) {
      if (this_mmd_process == 0) {
        // Serial
        store_md_simulations(md_simulations, scale_bridging_data);
      }
    } else {
      if (this_mmd_process <= n_qp) {
        // Parallel, one MPI process per quadrature point + rank 0
        p_store_md_simulations(md_simulations, scale_bridging_data);
      }
      MPI_Barrier(mmd_communicator);
    }
  }
}
} // namespace HMM

#endif
