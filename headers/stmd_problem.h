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
  	SymmetricTensor<2,dim> stress_from_hookes_law (SymmetricTensor<2,dim> strain);

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
		char locff[1024]; /*reaxff*/
		if (md_sim.force_field == "reax"){
			sprintf(locff, "%s/ffield.reax.2", md_sim.scripts_folder.c_str()); /*reaxff*/
		}
		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "g%d_%d", md_sim.material, md_sim.replica);

		char initdata[1024];
		sprintf(initdata, "%s/init.%s.bin", md_sim.output_folder.c_str(), mdstate);

		char straindata_last[1024];
		sprintf(straindata_last, "%s/last.%d.%s.dump", md_sim.output_folder.c_str(),
				md_sim.qp_id, mdstate);

		char straindata_time[1024];
		sprintf(straindata_time, "%s/%d.%d.%s.dump", md_sim.restart_folder.c_str(),
				md_sim.time_id, md_sim.qp_id, mdstate);

		char straindata_lcts[1024];
		sprintf(straindata_lcts, "%s/lcts.%d.%s.dump", md_sim.restart_folder.c_str(),
				md_sim.qp_id, mdstate);

		char homogdata_time[1024];
		sprintf(homogdata_time, "%s/%d.%d.%s.lammpstrj", md_sim.log_file.c_str(),
				md_sim.time_id, md_sim.qp_id, mdstate);

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
		sprintf(lmparg[4], "%s/log.stress_strain", md_sim.log_file.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

		// Passing location for output as variable
		sprintf(cline, "variable mdt string %d", md_sim.material); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", md_sim.log_file.c_str()); lammps_command(lmp,cline);
		if (md_sim.force_field == "reax"){
			sprintf(cline, "variable locf string %s", locff); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}
		
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
		std::ifstream ifile(straindata_last);
		if (ifile.good()){
			/*mdcout << "  specifically computed." << std::endl;*/
			ifile.close();

			if (md_sim.force_field == "reax") {
				sprintf(cline, "read_restart %s", initdata); lammps_command(lmp,cline); /*reaxff*/
				sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last); /*reaxff*/
				lammps_command(lmp,cline); /*reaxff*/
			}
			else if (md_sim.force_field == "opls") {
				sprintf(cline, "read_restart %s", straindata_last); /*opls*/
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

		// Number of timesteps in the MD simulation, enforcing at least one.
		int nts;
		nts = md_sim.strain.norm() / (md_sim.timestep_length * md_sim.strain_rate);
		nts = std::max(nts,1);
		

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
			sprintf(cline, "write_restart %s", straindata_last); /*opls*/
			lammps_command(lmp,cline); /*opls*/
		}
		else if (md_sim.force_field == "reax"){
			sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_last); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}

		if(md_sim.checkpoint){
			if (md_sim.force_field == "opls") {
				sprintf(cline, "write_restart %s", straindata_lcts); lammps_command(lmp,cline); /*opls*/
				sprintf(cline, "write_restart %s", straindata_time); lammps_command(lmp,cline); /*opls*/
			}
			else if (md_sim.force_field == "reax") {
				sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_lcts); lammps_command(lmp,cline); /*reaxff*/
				sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_time); lammps_command(lmp,cline); /*reaxff*/
			}
		}
		// close down LAMMPS
		delete lmp;

		/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;*/

		// Creating LAMMPS instance
		sprintf(lmparg[4], "%s/log.homogenization", md_sim.log_file.c_str());
		lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

		if (md_sim.force_field == "reax"){
			sprintf(cline, "variable locf string %s", locff); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}
        sprintf(cline, "variable loco string %s", md_sim.log_file.c_str()); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_sim.temperature); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", md_sim.scripts_folder.c_str(), "in.set.lammps");
		lammps_file(lmp,cfile);

		if (md_sim.force_field == "reax"){
			sprintf(cline, "read_restart %s", initdata); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
			sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/

		}
		else if (md_sim.force_field == "opls"){
			sprintf(cline, "read_restart %s", straindata_last); /*opls*/
			lammps_command(lmp,cline); /*opls*/
		}

		sprintf(cline, "variable dts equal %f", md_sim.timestep_length); lammps_command(lmp,cline);

		if(md_sim.output_homog){
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

		if(md_sim.output_homog){
			// Unetting dumping of atom positions
			sprintf(cline, "undump atom_dump"); lammps_command(lmp,cline);
		}

		// close down LAMMPS
		delete lmp;

		return md_sim.stress;
	}


	template <int dim>
  SymmetricTensor<2,dim> STMDProblem<dim>::stress_from_hookes_law (SymmetricTensor<2,dim> strain)
	{
		SymmetricTensor<2,dim> stress;

		SymmetricTensor<4,dim> stiffness;
    read_tensor<dim>("nanoscale_input/init.g0_1.stiff", stiffness);

		stress = stiffness * strain;
		return stress;
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
			std::cout << " \t" << md_sim.qp_id <<"-"<< md_sim.replica<<"-start" << std::flush;
		}
		MPI_Barrier(md_batch_communicator);

		if (approx_md_with_hookes_law == true){
			// this option is meant for testing
			md_sim.stress = stress_from_hookes_law(md_sim.strain);
			md_sim.stress_updated = true;
		}
		else {
			md_sim.stress = lammps_straining(md_sim);
			md_sim.stress_updated = true;
		}

		MPI_Barrier(md_batch_communicator);
		if(this_md_batch_process == 0)
		{
			std::cout << " \t" << md_sim.qp_id <<"-"<< md_sim.replica << std::flush;
		}
	}
}

#endif
