#ifndef STMD_PROBLEM_H
#define STMD_PROBLEM_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "library.h"
#include "atom.h"

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
//#include "boost/filesystem.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/mpi.h>

// Specifically built header files
#include "md_sim.h"
#include "read_write.h"
#include "stmd_sync.h"

//#include <boost/filesystem.hpp>

namespace HMM
{
using namespace dealii;
using namespace LAMMPS_NS;


template <int dim>
class STMDProblem
{
public:
	STMDProblem (MPI_Comm mdcomm, int pcolorr);
	~STMDProblem ();
	void strain (MDSim<dim>& md_sim, bool approx_md_with_hookes_law);

private:

	SymmetricTensor<2,dim> lammps_straining(MDSim<dim> md_sim);
	SymmetricTensor<2,dim> stress_from_hookes_law (SymmetricTensor<2,dim> strain, SymmetricTensor<4,dim> stiffness);
	void write_local_data (MDSim<dim> md_sim/*, SymmetricTensor<2,dim> stress_sample*/);

	MPI_Comm 							md_batch_communicator;
	const int 							md_batch_n_processes;
	const int 							this_md_batch_process;
	int 								md_batch_pcolor;

	ConditionalOStream 					mdcout;

};


template <int dim>
STMDProblem<dim>::STMDProblem (MPI_Comm mdcomm, int pcolor)
:
md_batch_communicator (mdcomm),
md_batch_n_processes (Utilities::MPI::n_mpi_processes(md_batch_communicator)),
this_md_batch_process (Utilities::MPI::this_mpi_process(md_batch_communicator)),
md_batch_pcolor (pcolor),
mdcout (std::cout,(this_md_batch_process == 0))
{}


template <int dim>
STMDProblem<dim>::~STMDProblem ()
{}


// The straining function is ran on every quadrature point which
// requires a stress_update. Since a quandrature point is only reached*
// by a subset of processes N, we should automatically see lammps be
// parallelized on the N processes.
template <int dim>
SymmetricTensor<2,dim> STMDProblem<dim>::lammps_straining (MDSim<dim> md_sim)
{
	bool store_log = true;
	if (md_sim.log_file == "none") store_log = false;

	if(store_log) mkdir(md_sim.log_file.c_str(), ACCESSPERMS);

	char locff[1024]; /*reaxff*/
	if (md_sim.force_field == "reax"){
		sprintf(locff, "%s/ffield.reax.2", md_sim.scripts_folder.c_str()); /*reaxff*/
	}
	// Name of nanostate binary files
	char mdstate[1024];
	sprintf(mdstate, "%s_%d", md_sim.matid.c_str(), md_sim.replica);

	char initdata[1024];
	sprintf(initdata, "%s/init.%s.bin", md_sim.output_folder.c_str(), mdstate);

	char homogdata_time[1024];
	if(store_log) {
		sprintf(homogdata_time, "%s/%s.%d.%s.lammpstrj", md_sim.log_file.c_str(),
				md_sim.time_id.c_str(), md_sim.qp_id, mdstate);
	}

	char straindata_lcts[1024];
	sprintf(straindata_lcts, "%s/lcts.%d.%s.dump", md_sim.restart_folder.c_str(),
			md_sim.qp_id, mdstate);

	// Find relevant dump file,
	// Check if this qp made a dump file of or it is branching from a previous qp
	char straindata_last_load[1024];
	char straindata_last_write[1024];
	if (md_sim.qp_id != md_sim.most_recent_qp_id){
		sprintf(straindata_last_load, "%s/last.%d.%s.dump", md_sim.output_folder.c_str(),
				md_sim.most_recent_qp_id, mdstate);
		sprintf(straindata_last_write, "%s/last.%d.%s.dump", md_sim.output_folder.c_str(),
						md_sim.qp_id, mdstate);
		// Check if the 'most_recent_qp_id' is still set as its default value,
		// if it is then there should not be any dumpfile to be found which will cause
		// the MD simulation should be performed using the initial position of the atoms,
		// else the dump file should exist
		std::ifstream ifile_most_recent(straindata_last_load);
		if(md_sim.most_recent_qp_id==std::numeric_limits<uint32_t>::max()){
			assert (ifile_most_recent.good() == false);
		}
		else{
			assert (ifile_most_recent.good() == true);
			ifile_most_recent.close();
		}
	}
	else {
		sprintf(straindata_last_load, "%s/last.%d.%s.dump", md_sim.output_folder.c_str(),
				md_sim.qp_id, mdstate);
		sprintf(straindata_last_write, "%s", straindata_last_load);
	}

	char cline[1024];
	char cfile[1024];

	// Specifying the command line options for screen and log output file
	int nargs = 5;
	char **lmparg = new char*[nargs];
	lmparg[0] = NULL;
	lmparg[1] = (char *) "-screen";
	lmparg[2] = (char *) "none";
	lmparg[3] = (char *) "-log";
	lmparg[4] = new char[1024];
	if(store_log) sprintf(lmparg[4], "%s/log.stress_strain", md_sim.log_file.c_str());
	else sprintf(lmparg[4], "none");

	// Creating LAMMPS instance
	LAMMPS *lmp = NULL;
	lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

	// Passing location for output as variable
	sprintf(cline, "variable mdt string %s", md_sim.matid.c_str()); lammps_command(lmp,cline);
	if(store_log) {sprintf(cline, "variable loco string %s", md_sim.log_file.c_str()); lammps_command(lmp,cline);}
	sprintf(cline, "variable locs string %s", md_sim.scripts_folder.c_str()); lammps_command(lmp,cline);

	// Setting testing temperature
	sprintf(cline, "variable tempt equal %f", md_sim.temperature); lammps_command(lmp,cline);

	// Setting general parameters for LAMMPS independentely of what will be
	// tested on the sample next.

	sprintf(cfile, "%s/%s", md_sim.scripts_folder.c_str(), "in.set.lammps");
	lammps_file(lmp,cfile);

	/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Compute current state data...       " << std::endl;*/

	// Check if a previous state has already been computed specifically for
	// this quadrature point, otherwise use the initial state (which is the
	// last state of this quadrature point)
	/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... from previous state data...   " << std::flush;*/

	// Check the presence of a dump file to restart from
	// (either from current or previous ID_to_get_results_from)
	std::ifstream ifile(straindata_last_load);
	if (ifile.good()){
		/*mdcout << "  specifically computed." << std::endl;*/
		ifile.close();

		if (md_sim.force_field == "reax") {
			sprintf(cline, "read_restart %s", initdata); lammps_command(lmp,cline); /*reaxff*/
			sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last_load); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}
		else if (md_sim.force_field == "opls") {
			sprintf(cline, "read_restart %s", straindata_last_load); /*opls*/
			lammps_command(lmp,cline); /*opls*/
		}

		sprintf(cline, "print 'specifically computed'"); lammps_command(lmp,cline);
	}
	else{
		/*mdcout << "  initially computed." << std::endl;*/
		sprintf(cline, "read_restart %s", initdata);
		lammps_command(lmp,cline);
		sprintf(cline, "print 'initially computed'"); lammps_command(lmp,cline);
	}


	// Query box dimensions
	char vdir[1024];
	std::vector<double> lbdim (dim);
	sprintf(cline, "variable ll1 equal lx"); lammps_command(lmp,cline);
	sprintf(cline, "variable ll2 equal ly"); lammps_command(lmp,cline);
	sprintf(cline, "variable ll3 equal lz"); lammps_command(lmp,cline);
	for(unsigned int i=0;i<dim;i++){
		sprintf(vdir, "ll%d",i+1);
		lbdim[i] = *((double *) lammps_extract_variable(lmp,vdir,NULL));
	}

	// Correction of strain tensor with actual box dimensions
	for (unsigned int i=0; i<dim; i++){
		md_sim.strain[i][i] /= lbdim[i];
		md_sim.strain[i][(i+1)%dim] /= lbdim[(i+2)%dim];
	}

	// Number of timesteps in the MD simulation, rounding to nearest 10, enforcing at least 10
	//int nts = std::max(int(std::ceil(md_sim.strain.norm()/(md_sim.timestep_length*md_sim.strain_rate)/10)*10),1);
	int nts;
	double strain_time = md_sim.strain.norm() / md_sim.strain_rate;
	nts = std::ceil( (strain_time/md_sim.timestep_length) /10.0) * 10;// rounded to the nearest 10
	nts = std::max(nts,10); // check that its not 0


	sprintf(cline, "variable dts equal %f", md_sim.timestep_length); lammps_command(lmp,cline);
	sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

	for(unsigned int k=0;k<dim;k++)
		for(unsigned int l=k;l<dim;l++)
		{
			sprintf(cline, "variable ceeps_%d%d equal %.6e", k, l,
					md_sim.strain[k][l]/(nts*md_sim.timestep_length));
			lammps_command(lmp,cline);
		}

	// Run the NEMD simulations of the strained box
	/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... reading and executing in.strain.lammps.       " << std::endl;*/
	sprintf(cfile, "%s/%s", md_sim.scripts_folder.c_str(), "in.strain.lammps");
	lammps_file(lmp,cfile);

	/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;*/
	// Save data to specific file for this quadrature point
	if (md_sim.force_field == "opls"){
		sprintf(cline, "write_restart %s", straindata_last_write); /*opls*/
		lammps_command(lmp,cline); /*opls*/
	}
	else if (md_sim.force_field == "reax"){
		sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_last_write); /*reaxff*/
		lammps_command(lmp,cline); /*reaxff*/
	}

	if(md_sim.checkpoint){
		if (md_sim.force_field == "opls") {
			sprintf(cline, "write_restart %s", straindata_lcts); lammps_command(lmp,cline); /*opls*/
		}
		else if (md_sim.force_field == "reax") {
			sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_lcts); lammps_command(lmp,cline); /*reaxff*/
		}
	}
	// close down LAMMPS
	delete lmp;

	/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;*/

	// Creating LAMMPS instance
	if(store_log) sprintf(lmparg[4], "%s/log.homogenization", md_sim.log_file.c_str());
	else sprintf(lmparg[4], "none");
	lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

	if(store_log) {sprintf(cline, "variable loco string %s", md_sim.log_file.c_str()); lammps_command(lmp,cline);}
	sprintf(cline, "variable locs string %s", md_sim.scripts_folder.c_str()); lammps_command(lmp,cline);

	// Setting testing temperature
	sprintf(cline, "variable tempt equal %f", md_sim.temperature); lammps_command(lmp,cline);

	// Setting general parameters for LAMMPS independentely of what will be
	// tested on the sample next.
	sprintf(cfile, "%s/%s", md_sim.scripts_folder.c_str(), "in.set.lammps");
	lammps_file(lmp,cfile);

	if (md_sim.force_field == "reax"){
		sprintf(cline, "read_restart %s", initdata); /*reaxff*/
		lammps_command(lmp,cline); /*reaxff*/
		sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last_write); /*reaxff*/
		lammps_command(lmp,cline); /*reaxff*/

	}
	else if (md_sim.force_field == "opls"){
		sprintf(cline, "read_restart %s", straindata_last_write); /*opls*/
		lammps_command(lmp,cline); /*opls*/
	}

	sprintf(cline, "reset_timestep 0"); lammps_command(lmp,cline);

	sprintf(cline, "variable dts equal %f", md_sim.timestep_length); lammps_command(lmp,cline);

	if(md_sim.output_homog && store_log){
		// Setting dumping of atom positions for post analysis of the MD simulation
		// DO NOT USE CUSTOM DUMP: WRONG ATOM POSITIONS...
		sprintf(cline, "dump atom_dump all atom %d %s", 1, homogdata_time); lammps_command(lmp,cline);
	}

	// Compute the secant stiffness tensor at the given stress/strain state
	sprintf(cline, "variable locbe string %s/%s", md_sim.scripts_folder.c_str(), "ELASTIC");
	lammps_command(lmp,cline);

	// Set sampling and straining time-lengths
	sprintf(cline, "variable nssample0 equal %d", md_sim.nsteps_sample); lammps_command(lmp,cline);
	sprintf(cline, "variable nssample  equal %d", md_sim.nsteps_sample); lammps_command(lmp,cline);

	// Using a routine based on the example ELASTIC/ to compute the stress tensor
	sprintf(cfile, "%s/%s", md_sim.scripts_folder.c_str(), "ELASTIC/in.homogenization.lammps");
	lammps_file(lmp,cfile);

	// Filling 3x3 stress tensor and conversion from ATM to Pa
	// Useless at the moment, since it cannot be used in the Newton-Raphson algorithm.
	// The MD evaluated stress is flucutating too much (few MPa), therefore prevents
	// the iterative algorithm to converge...
	for(unsigned int k=0;k<dim;k++)
		for(unsigned int l=k;l<dim;l++)
		{
			char vcoef[1024];
			sprintf(vcoef, "pp%d%d", k+1, l+1);
			md_sim.stress[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*(-1.0)*1.01325e+05;
		}

	// (stress distribution) Retrieve a vector of stress tensors (vectorised stress tensor)
	// this is commented because I cannot extract data from stress time series correctly (wrong ordering)
	/*std::vector <SymmetricTensor<2,dim>> stress_dist;
	for(unsigned int k=0;k<md_sim.nsteps_sample;k++){
		SymmetricTensor<2,dim>  stress_sample;
		for(unsigned int l=0;l<2*dim;l++)
		{
			double *dptr = (double *) lammps_extract_fix(lmp, "stress_series", 0, 2, k+1, l+1);
			if (l<dim){
				stress_sample[l][l] = *dptr;
			}
			else if (l==dim) stress_sample[0][1] = *dptr;
			else if (l==dim+1) stress_sample[0][dim-1] = *dptr;
			else if (dim>2 && l==2*dim-1) stress_sample[1][dim-1] = *dptr;
			lammps_free(dptr);
		}
		stress_dist.push_back(stress_sample);
	}
	write_local_data(md_sim, stress_sample);*/

	if(md_sim.output_homog){
		// Unetting dumping of atom positions
		sprintf(cline, "undump atom_dump"); lammps_command(lmp,cline);
	}

	// close down LAMMPS
	delete lmp;

	if(store_log) {
		// Clean "nanoscale_logs" of the finished timestep
		char command[1024];
		sprintf(command, "rm -rf %s", md_sim.log_file.c_str());
		//std::cout<< "Logfile "<< md_simulation.log_file <<std::endl;
		int ret = system(command);
		if (ret!=0){
			std::cout << "Failed removing the log files of the MD simulation: " << md_sim.log_file << std::endl;
		}
		//boost::filesystem::remove_all(md_sim.log_file.c_str());
	}
	return md_sim.stress;
}


template <int dim>
SymmetricTensor<2,dim> STMDProblem<dim>::stress_from_hookes_law (SymmetricTensor<2,dim> strain, SymmetricTensor<4,dim> stiffness)
{
	SymmetricTensor<2,dim> stress;
	stress = stiffness * strain;
	return stress;
}

template <int dim>
void STMDProblem<dim>::write_local_data(md_sim/*, stress_sample*/)
{
	// (stress distribution) Append molecular model data file
	if(this_md_batch_process == 0){

	// Initialization of the molecular data file
	char filename[1024]; sprintf(filename, "%s/mddata_qpid%d_repl%d.csv", md_sim.output_folder.c_str(), md_sim.qp_id, md_sim.replica);
	std::ofstream  ofile(filename, std::ios_base::app);
	long cursor_position = ofile.tellp();

	// writing the header of the file (if file is empty)
	if (cursor_position == 0){
		ofile << "qp_id,material_id,homog_time_id,temperature,strain_rate,force_field,replica_id";
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				ofile << "," << "strain_" << k << l;
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
				ofile << "," << "stress_" << k << l;
		ofile << std::endl;
	}

	// writing averaged data over homogenisation time steps
	ofile << md_sim.qp_id
			 << "," << md_sim.matid
			 << "," << "averaged"
			 << "," << md_sim.temperature
			 << "," << md_sim.strain_rate
			 << "," << md_sim.force_field
			 << "," << md_sim.replica;
	for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k;l<dim;l++){
			  ofile << "," << std::setprecision(16) << md_sim.strain[k][l];
		  }
	for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k;l<dim;l++){
			  ofile << "," << std::setprecision(16) << md_sim.stress[k][l];
		  }
	ofile << std::endl;
		
	// writing current time data
	// this is commented because I cannot extract data from stress time series correctly (wrong ordering)
	/*for(unsigned int t=0;t<md_sim.nsteps_sample;t++){
	   ofile << md_sim.qp_id
			 << "," << md_sim.matid
			 << "," << md_sim.time_id
			 << "," << md_sim.temperature
			 << "," << md_sim.strain_rate
			 << "," << md_sim.force_field
			 << "," << md_sim.replica;
	   for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k;l<dim;l++){
			  ofile << "," << std::setprecision(16) << md_sim.strain[k][l];
		  }
	   for(unsigned int k=0;k<dim;k++)
		  for(unsigned int l=k;l<dim;l++){
			  ofile << "," << std::setprecision(16) << stress_dist[t][k][l];
		  }
	   ofile << std::endl;
	}*/
}

template <int dim>
void STMDProblem<dim>::strain (MDSim<dim>& md_sim, bool approx_md_with_hookes_law)
{

	if (md_sim.force_field != "opls" && md_sim.force_field != "reax"){
		std::cerr << "Error: Force field is " << md_sim.force_field
				<< " but only 'opls' and 'reax' are implemented... "
				<< std::endl;
		exit(1);
	}

	// Then the lammps function instanciates lammps, starting from an initial
	// microstructure and applying the complete new_strain or starting from
	// the microstructure at the old_strain and applying the difference between
	// the new_ and _old_strains, returns the new_stress state.
	if(this_md_batch_process == 0)
	{
		std::cout << " \t" << md_sim.qp_id <<"-"<< md_sim.replica<<"-start" << std::endl << std::flush;
	}
	MPI_Barrier(md_batch_communicator);

	if (approx_md_with_hookes_law == true){
		// this option is meant for testing
		md_sim.stress = stress_from_hookes_law(md_sim.strain, md_sim.stiffness);
		md_sim.stress_updated = true;
	}
	else {
		md_sim.stress = lammps_straining(md_sim);
		md_sim.stress_updated = true;
	}
	
	write_local_data(mdsim);

	MPI_Barrier(md_batch_communicator);
	if(this_md_batch_process == 0)
	{
		std::cout << " \t" << md_sim.qp_id <<"-"<< md_sim.replica << std::endl << std::flush;
	}
}
}

#endif
