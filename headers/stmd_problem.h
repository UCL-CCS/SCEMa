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
//#include "md_sim.h"
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
		STMDProblem (MPI_Comm mdcomm, int pcolor);
		~STMDProblem ();
		void strain (MDSim<dim> md_sim);
		//void strain (std::string cid, std::string 	tid, std::string cmat,
		//		  std::string slocout, std::string slocres, std::string llochom,
		//		  std::string qplogloc, std::string scrloc,
		//		  std::string strainif, std::string stressof,
		//		  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
		//		  double mdss, std::string mdff, bool outhom, bool checksav);

	private:

		void lammps_straining();

		MPI_Comm 							md_batch_communicator;
		const int 							md_batch_n_processes;
		const int 							this_md_batch_process;
		int 								md_batch_pcolor;

		ConditionalOStream 					mdcout;

		SymmetricTensor<2,dim> 				loc_rep_strain;
		SymmetricTensor<2,dim> 				loc_rep_stress;

		std::string 						cellid;
		std::string 						timeid;
		std::string 						cellmat;

		std::string 						statelocout;
		std::string 						statelocres;
		std::string 						loglochom;
		std::string 						qpreplogloc;
		std::string 						scriptsloc;

		SymmetricTensor<2,dim>	strain_in;
		SymmetricTensor<2,dim>	stress_out;

		unsigned int 						repl;

		double								md_timestep_length;
		double								md_temperature;
		unsigned int 						md_nsteps_sample;
		double								md_strain_rate;
		std::string							md_force_field;

		bool								output_homog;
		bool								checkpoint_save;

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
	void STMDProblem<dim>::lammps_straining ()
	{
		char locff[1024]; /*reaxff*/
		if (md_force_field == "reax"){
			sprintf(locff, "%s/ffield.reax.2", scriptsloc.c_str()); /*reaxff*/
		}
		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "%s_%d", cellmat.c_str(), repl);

		char initdata[1024];
		sprintf(initdata, "%s/init.%s.bin", statelocout.c_str(), mdstate);

		char straindata_last[1024];
		sprintf(straindata_last, "%s/last.%s.%s.dump", statelocout.c_str(),
				cellid.c_str(), mdstate);
		// sprintf(straindata_last, "%s/last.%s.%s.bin", statelocout.c_str(), cellid, mdstate);

		char straindata_time[1024];
		sprintf(straindata_time, "%s/%s.%s.%s.dump", statelocres.c_str(),
				timeid.c_str(), cellid.c_str(), mdstate);
		// sprintf(straindata_lcts, "%s/lcts.%s.%s.bin", statelocres.c_str(), cellid, mdstate);

		char straindata_lcts[1024];
		sprintf(straindata_lcts, "%s/lcts.%s.%s.dump", statelocres.c_str(),
				cellid.c_str(), mdstate);
		// sprintf(straindata_lcts, "%s/lcts.%s.%s.bin", statelocres.c_str(), cellid, mdstate);

		char homogdata_time[1024];
		sprintf(homogdata_time, "%s/%s.%s.%s.lammpstrj", loglochom.c_str(),
				timeid.c_str(), cellid.c_str(), mdstate);

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
		sprintf(lmparg[4], "%s/log.stress_strain", qpreplogloc.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

		// Passing location for output as variable
		sprintf(cline, "variable mdt string %s", cellmat.c_str()); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", qpreplogloc.c_str()); lammps_command(lmp,cline);
		if (md_force_field == "reax"){
			sprintf(cline, "variable locf string %s", locff); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_temperature); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.

		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.set.lammps");
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

			if (md_force_field == "reax") {
				sprintf(cline, "read_restart %s", initdata); lammps_command(lmp,cline); /*reaxff*/
				sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last); /*reaxff*/
				lammps_command(lmp,cline); /*reaxff*/
			}
			else if (md_force_field == "opls") {
				sprintf(cline, "read_restart %s", straindata_last); /*opls*/
				lammps_command(lmp,cline); /*opls*/
			}

			sprintf(cline, "print 'specifically computed'"); lammps_command(lmp,cline);
		}
		else{
			/*mdcout << "  initially computed." << std::endl;*/
			std::cout<<"else "<<std::endl;
			std::cout<<"init "<<initdata<<std::endl;
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
			loc_rep_strain[i][i] /= lbdim[i];
			loc_rep_strain[i][(i+1)%dim] /= lbdim[(i+2)%dim];
		}

		// Number of timesteps in the MD simulation, enforcing at least one.
		int nts = std::max(int(std::ceil(loc_rep_strain.norm()/(md_timestep_length*md_strain_rate)/10)*10),1);

		sprintf(cline, "variable dts equal %f", md_timestep_length); lammps_command(lmp,cline);
		sprintf(cline, "variable nts equal %d", nts); lammps_command(lmp,cline);

		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				sprintf(cline, "variable ceeps_%d%d equal %.6e", k, l, loc_rep_strain[k][l]/(nts*md_timestep_length));
				lammps_command(lmp,cline);
			}

		// Run the NEMD simulations of the strained box
		/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "   ... reading and executing in.strain.lammps.       " << std::endl;*/
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.strain.lammps");
		lammps_file(lmp,cfile);

		/*mdcout << "               "
				<< "(MD - " << timeid <<"."<< cellid << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;*/
		// Save data to specific file for this quadrature point
		if (md_force_field == "opls"){
			sprintf(cline, "write_restart %s", straindata_last); /*opls*/
			lammps_command(lmp,cline); /*opls*/
		}
		else if (md_force_field == "reax"){
			sprintf(cline, "write_dump all custom %s id type xs ys zs vx vy vz ix iy iz", straindata_last); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}

		if(checkpoint_save){
			if (md_force_field == "opls") {
				sprintf(cline, "write_restart %s", straindata_lcts); lammps_command(lmp,cline); /*opls*/
				sprintf(cline, "write_restart %s", straindata_time); lammps_command(lmp,cline); /*opls*/
			}
			else if (md_force_field == "reax") {
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
		sprintf(lmparg[4], "%s/log.homogenization", qpreplogloc.c_str());
		lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

		if (md_force_field == "reax"){
			sprintf(cline, "variable locf string %s", locff); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}
        sprintf(cline, "variable loco string %s", qpreplogloc.c_str()); lammps_command(lmp,cline);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_temperature); lammps_command(lmp,cline);

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.set.lammps");
		lammps_file(lmp,cfile);

		if (md_force_field == "reax"){
			sprintf(cline, "read_restart %s", initdata); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
			sprintf(cline, "rerun %s dump x y z vx vy vz ix iy iz box yes scaled yes wrapped yes format native", straindata_last); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/

		}
		else if (md_force_field == "opls"){
			sprintf(cline, "read_restart %s", straindata_last); /*opls*/
			lammps_command(lmp,cline); /*opls*/
		}

		sprintf(cline, "variable dts equal %f", md_timestep_length); lammps_command(lmp,cline);

		if(output_homog){
			// Setting dumping of atom positions for post analysis of the MD simulation
			// DO NOT USE CUSTOM DUMP: WRONG ATOM POSITIONS...
			sprintf(cline, "dump atom_dump all atom %d %s", 1, homogdata_time); lammps_command(lmp,cline);
		}

		// Compute the secant stiffness tensor at the given stress/strain state
		sprintf(cline, "variable locbe string %s/%s", scriptsloc.c_str(), "ELASTIC");
		lammps_command(lmp,cline);

		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", md_nsteps_sample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", md_nsteps_sample); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress tensor
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "ELASTIC/in.homogenization.lammps");
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
				loc_rep_stress[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*(-1.0)*1.01325e+05;
			}

		if(output_homog){
			// Unetting dumping of atom positions
			sprintf(cline, "undump atom_dump"); lammps_command(lmp,cline);
		}

		// close down LAMMPS
		delete lmp;
	}



	template <int dim>
  void STMDProblem<dim>::strain (MDSim<dim> md_sim)
	//void STMDProblem<dim>::strain (std::string cid, std::string 	tid, std::string cmat,
	//						  std::string slocout, std::string slocres, std::string llochom,
	//						  std::string qplogloc, std::string scrloc,
	//						  std::string strainif, std::string stressof,
	//						  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
	//						  double mdss, std::string mdff, bool outhom, bool checksav)
	{
		cellid = std::to_string(md_sim.qp_id);
		timeid = md_sim.time_id;
		char temp[1024];
		sprintf(temp, "g%d", md_sim.material);
		cellmat = temp;

		statelocout = md_sim.output_file;
		statelocres = md_sim.restart_file;
		loglochom = md_sim.log_file;
		qpreplogloc = md_sim.log_file;
		scriptsloc = "./lammps_scripts_opls/";

		strain_in = md_sim.strain;
		stress_out = md_sim.stress;

		repl = md_sim.replica;

		md_timestep_length 	= md_sim.timestep_length;
		md_temperature 			= md_sim.temperature;
		md_nsteps_sample 		= md_sim.nsteps_sample;
		md_strain_rate 			= md_sim.strain_rate;
		md_force_field 			=	md_sim.force_field;

		output_homog = md_sim.output_homog;
		checkpoint_save = md_sim.checkpoint;

		if (md_force_field != "opls" && md_force_field != "reax"){
			std::cerr << "Error: Force field is " << md_force_field
					  << " but only 'opls' and 'reax' are implemented... "
					  << std::endl;
			exit(1);
		}
/*
		std::cout << "TEST MD INPUT" << std::endl;
		// loc_rep_strain, loc_rep_strain are tensors not not filenames
		std::cout<< 						"1 "<<cellid<<std::endl;
		std::cout <<						"2 "<<timeid<<std::endl;
		std::cout 	<<					"3 "<<cellmat<<std::endl;

		std::cout 		<<				"4 "<<statelocout<<std::endl;
		std::cout 			<<			"5 "<<statelocres<<std::endl;
		std::cout 				<<		"6 "<<loglochom<<std::endl;
		std::cout 					<<	"7 "<<qpreplogloc<<std::endl;
		std::cout 						<<"8 "<<scriptsloc<<std::endl;

		//SymmetricTensor<2,dim>	strain_in;
		//SymmetricTensor<2,dim>	stress_out;

		unsigned int 						repl;

		double								md_timestep_length;
		double								md_temperature;
		unsigned int 						md_nsteps_sample;
		double								md_strain_rate;
		std::string							md_force_field;

		bool								output_homog;
		bool								checkpoint_save;

		std::cout << "FINISH MD TEST"<<std::endl;*/
		// Argument of the MD simulation: strain to apply
		//sprintf(filename, "%s/last.%s.%d.upstrain", macrostatelocout.c_str(), cellid, repl);
		//read_tensor<dim>(straininputfile.c_str(), loc_rep_strain);
		loc_rep_strain = md_sim.strain;
		// Then the lammps function instanciates lammps, starting from an initial
		// microstructure and applying the complete new_strain or starting from
		// the microstructure at the old_strain and applying the difference between
		// the new_ and _old_strains, returns the new_stress state.
		lammps_straining();

		if(this_md_batch_process == 0)
		{
			std::cout << " \t" << cellid <<"-"<< repl << std::flush;

			//sprintf(filename, "%s/last.%s.%d.stress", macrostatelocout.c_str(), cellid, repl);
			//write_tensor<dim>(stressoutputfile.c_str(), loc_rep_stress);
		}
	}
}

#endif
