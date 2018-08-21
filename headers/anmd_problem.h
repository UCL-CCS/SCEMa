#ifndef ANMD_PROBLEM_H
#define ANMD_PROBLEM_H

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
#include "read_write.h"

namespace HMM
{
	using namespace dealii;
	using namespace LAMMPS_NS;


	template <int dim>
	class ANMDProblem
	{
	public:
		ANMDProblem (MPI_Comm mdcomm, int pcolor);
		~ANMDProblem ();
		void analyse (std::string cid, std::string 	tid, std::string cmat,
				  std::string slocout, std::string slocres,
				  std::string qplogloc, std::string scrloc,
				  unsigned int rep, double mdts, double mdtem,
				  double mdss, std::string mdff);

	private:

		void lammps_analyse();

		MPI_Comm 							md_batch_communicator;
		const int 							md_batch_n_processes;
		const int 							this_md_batch_process;
		int 								md_batch_pcolor;

		ConditionalOStream 					mdcout;

		std::string 						cellid;
		std::string 						timeid;
		std::string 						cellmat;

		std::string 						statelocout;
		std::string 						statelocres;
		std::string 						qpreplogloc;
		std::string 						scriptsloc;

		unsigned int 						repl;

		double								md_timestep_length;
		double								md_temperature;
		unsigned int 						md_nsteps_sample;
		std::string							md_force_field;

	};



	template <int dim>
	ANMDProblem<dim>::ANMDProblem (MPI_Comm mdcomm, int pcolor)
	:
		md_batch_communicator (mdcomm),
		md_batch_n_processes (Utilities::MPI::n_mpi_processes(md_batch_communicator)),
		this_md_batch_process (Utilities::MPI::this_mpi_process(md_batch_communicator)),
		md_batch_pcolor (pcolor),
		mdcout (std::cout,(this_md_batch_process == 0))
	{}



	template <int dim>
	ANMDProblem<dim>::~ANMDProblem ()
	{}



	// The straining function is ran on every quadrature point which
	// requires a stress_update. Since a quandrature point is only reached*
	// by a subset of processes N, we should automatically see lammps be
	// parallelized on the N processes.
	template <int dim>
	void ANMDProblem<dim>::lammps_analyse ()
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
		sprintf(straindata_last, "%s/%s.%s.%s.dump", statelocres.c_str(), timeid.c_str(),
				cellid.c_str(), mdstate);
		// sprintf(straindata_last, "%s/last.%s.%s.bin", statelocout.c_str(), cellid, mdstate);

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
		sprintf(lmparg[4], "%s/log.homogenization", qpreplogloc.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
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

		// Compute the secant stiffness tensor at the given stress/strain state
		sprintf(cline, "variable locbe string %s/%s", scriptsloc.c_str(), "ELASTIC");
		lammps_command(lmp,cline);

		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", md_nsteps_sample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", md_nsteps_sample); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress tensor
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "ELASTIC/in.analyse.lammps");
		lammps_file(lmp,cfile);

		// close down LAMMPS
		delete lmp;
	}



	template <int dim>
	void ANMDProblem<dim>::analyse (std::string cid, std::string 	tid, std::string cmat,
							  std::string slocout, std::string slocres,
							  std::string qplogloc, std::string scrloc,
							  unsigned int rep, double mdts, double mdtem,
							  double mdnss, std::string mdff)
	{
		cellid = cid;
		timeid = tid;
		cellmat = cmat;

		statelocres = slocout;
		statelocres = slocres;
		qpreplogloc = qplogloc;
		scriptsloc = scrloc;

		repl = rep;

		md_timestep_length = mdts;
		md_temperature = mdtem;
		md_nsteps_sample = mdnss;
		md_force_field = mdff;

		if (md_force_field != "opls" && md_force_field != "reax"){
			std::cerr << "Error: Force field is " << md_force_field
					  << " but only 'opls' and 'reax' are implemented... "
					  << std::endl;
			exit(1);
		}

		// Then the lammps function instanciates lammps, starting from an initial
		// microstructure and applying the complete new_strain or starting from
		// the microstructure at the old_strain and applying the difference between
		// the new_ and _old_strains, returns the new_stress state.
		lammps_analyse();
	}
}

#endif
