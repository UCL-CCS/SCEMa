#ifndef EQMD_PROBLEM_H
#define EQMD_PROBLEM_H

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
	class EQMDProblem
	{
	public:
		EQMDProblem (MPI_Comm mdcomm, int pcolor);
		~EQMDProblem ();

		void equil (std::string cmat,
				  std::string slocin,
				  std::string qplogloc, std::string scrloc,
				  std::string lengthof, std::string stressof, std::string stiffof, std::string systeof,
				  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
				  unsigned int mdnse, double mdss, double mdsa, std::string mdff);

	private:

		void lammps_equilibration();

		MPI_Comm 							md_batch_communicator;
		const int 							md_batch_n_processes;
		const int 							this_md_batch_process;
		int 								md_batch_pcolor;

		ConditionalOStream 					mdcout;

		Tensor<1,dim>						loc_rep_length;
		SymmetricTensor<2,dim> 				loc_rep_stress;
		SymmetricTensor<4,dim> 				loc_rep_stiff;

		std::string 						cellmat;

		std::string 						statelocin;
		std::string 						qpreplogloc;
		std::string 						scriptsloc;

		std::string 						lengthoutputfile;
		std::string 						stressoutputfile;
		std::string 						stiffoutputfile;
		std::string 						systemoutputfile;

		unsigned int 						repl;

		double								md_timestep_length;
		double								md_temperature;
		unsigned int 						md_nsteps_sample;
		unsigned int 						md_nsteps_equil;
		double								md_strain_rate;
		double								md_strain_ampl;
		std::string							md_force_field;

	};



	template <int dim>
	EQMDProblem<dim>::EQMDProblem (MPI_Comm mdcomm, int pcolor)
	:
		md_batch_communicator (mdcomm),
		md_batch_n_processes (Utilities::MPI::n_mpi_processes(md_batch_communicator)),
		this_md_batch_process (Utilities::MPI::this_mpi_process(md_batch_communicator)),
		md_batch_pcolor (pcolor),
		mdcout (std::cout,(this_md_batch_process == 0))
	{}



	template <int dim>
	EQMDProblem<dim>::~EQMDProblem ()
	{}






	// The initiation, namely the preparation of the data from which will
	// be ran the later tests at every quadrature point, should be ran on
	// as many processes as available, since it will be the only on going
	// task at the time it will be called.
	template <int dim>
	void EQMDProblem<dim>::lammps_equilibration ()
	{
		char locff[1024]; /*reaxff*/
		if (md_force_field == "reax"){
			sprintf(locff, "%s/ffield.reax.2", scriptsloc.c_str()); /*reaxff*/
		}

		// Name of nanostate binary files
		char mdstate[1024];
		sprintf(mdstate, "%s_%d", cellmat.c_str(), repl);

		char locdata[1024];
		sprintf(locdata, "%s/%s.data", statelocin.c_str(), mdstate);

		char cfile[1024];
		char cline[1024];
		char sfile[1024];

		// Specifying the command line options for screen and log output file
		int nargs = 5;
		char **lmparg = new char*[nargs];
		lmparg[0] = NULL;
		lmparg[1] = (char *) "-screen";
		lmparg[2] = (char *) "none";
		lmparg[3] = (char *) "-log";
		lmparg[4] = new char[1024];
		sprintf(lmparg[4], "%s/log.heatup_cooldown", qpreplogloc.c_str());

		// Creating LAMMPS instance
		LAMMPS *lmp = NULL;
		lmp = new LAMMPS(nargs,lmparg,md_batch_communicator);

		// Setting run parameters
		sprintf(cline, "variable dts equal %f", md_timestep_length); lammps_command(lmp,cline);
		sprintf(cline, "variable nsinit equal %d", md_nsteps_equil); lammps_command(lmp,cline);

		// Passing location for input and output as variables
		sprintf(cline, "variable mdt string %s", cellmat.c_str()); lammps_command(lmp,cline);
		sprintf(cline, "variable locd string %s", locdata); lammps_command(lmp,cline);
		sprintf(cline, "variable loco string %s", qpreplogloc.c_str()); lammps_command(lmp,cline);

		if (md_force_field == "reax"){
			sprintf(cline, "variable locf string %s", locff); /*reaxff*/
			lammps_command(lmp,cline); /*reaxff*/
		}

		// Setting general parameters for LAMMPS independentely of what will be
		// tested on the sample next.
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.set.lammps"); lammps_file(lmp,cfile);

		// Setting testing temperature
		sprintf(cline, "variable tempt equal %f", md_temperature); lammps_command(lmp,cline);
		sprintf(cline, "variable sseed equal 1234"); lammps_command(lmp,cline);

		// Check if 'init.PE.bin' has been computed already
		sprintf(sfile, "%s", systemoutputfile.c_str());
		bool state_exists = file_exists(sfile);

		if (!state_exists)
		{
			mdcout << "(MD - init - type " << cellmat << " - repl " << repl << ") "
					<< "Compute state data...       " << std::endl;
			// Compute initialization of the sample which minimizes the free energy,
			// heat up and finally cool down the sample.
			sprintf(cfile, "%s/%s", scriptsloc.c_str(), "in.init.lammps"); lammps_file(lmp,cfile);
		}
		else
		{
			mdcout << "(MD - init - type " << cellmat << " - repl " << repl << ") "
					<< "Reuse of state data...       " << std::endl;
			// Reload from previously computed initial preparation (minimization and
			// heatup/cooldown), this option shouldn't remain, as in the first step the
			// preparation should always be computed.
			sprintf(cline, "read_restart %s", systemoutputfile.c_str()); lammps_command(lmp,cline);
		}

		// Storing initial dimensions after initiation
		char lname[1024];
		sprintf(lname, "lxbox0");
		sprintf(cline, "variable tmp equal 'lx'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		loc_rep_length[0] = *((double *) lammps_extract_variable(lmp,lname,NULL));
		sprintf(lname, "lybox0");
		sprintf(cline, "variable tmp equal 'ly'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		loc_rep_length[1] = *((double *) lammps_extract_variable(lmp,lname,NULL));
		sprintf(lname, "lzbox0");
		sprintf(cline, "variable tmp equal 'lz'"); lammps_command(lmp,cline);
		sprintf(cline, "variable %s equal ${tmp}", lname); lammps_command(lmp,cline);
		loc_rep_length[2] = *((double *) lammps_extract_variable(lmp,lname,NULL));

		// Saving nanostate at the end of initiation
		mdcout << "(MD - init - type " << cellmat << " - repl " << repl << ") "
				<< "Saving state data...       " << std::endl;
		sprintf(cline, "write_restart %s", systemoutputfile.c_str()); lammps_command(lmp,cline);

		mdcout << "(MD - init - type " << cellmat << " - repl " << repl << ") "
				<< "Homogenization of stiffness and stress using in.elastic.lammps...       " << std::endl;

		// Compute secant stiffness operator and initial stresses
		sprintf(cline, "variable locbe string %s/%s", scriptsloc.c_str(), "ELASTIC");
		lammps_command(lmp,cline);

		// Set sampling and straining time-lengths
		sprintf(cline, "variable nssample0 equal %d", md_nsteps_sample); lammps_command(lmp,cline);
		sprintf(cline, "variable nssample  equal %d", md_nsteps_sample); lammps_command(lmp,cline);

		// number of timesteps for straining
		int nsstrain = std::ceil(md_strain_ampl/(md_timestep_length*md_strain_rate)/10)*10;

		// For v_sound_PE = 2000 m/s, l_box=8nm, strain_perturbation=0.005, and dts=2.0fs
		// the min number of straining steps is 10
		sprintf(cline, "variable nsstrain  equal %d", nsstrain); lammps_command(lmp,cline);

		// Set strain perturbation amplitude
		sprintf(cline, "variable up equal %f", md_strain_ampl); lammps_command(lmp,cline);

		// Using a routine based on the example ELASTIC/ to compute the stress tensor
		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "ELASTIC/in.homogenization.lammps");
		lammps_file(lmp,cfile);

		// Filling 3x3 stress tensor and conversion from ATM to Pa
		// Useless at the moment, since it cannot be used in the Newton-Raphson algorithm.
		// The MD evaluated stress is flucutating too much (few MPa), therefore prevents
		// the iterative algorithm to converge...
		for(unsigned int k=0;k<dim;k++){
			for(unsigned int l=k;l<dim;l++)
			{
				char vcoef[1024];
				sprintf(vcoef, "pp%d%d", k+1, l+1);
				loc_rep_stress[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*(-1.0)*1.01325e+05;
			}
		}

		// Using a routine based on the example ELASTIC/ to compute the stiffness tensor
		for(unsigned int k=0;k<dim;k++)
			for(unsigned int l=k;l<dim;l++)
			{
				sprintf(cline, "variable eeps_%d%d equal %.6e", k, l, 0.0);
				lammps_command(lmp,cline);
			}

		sprintf(cfile, "%s/%s", scriptsloc.c_str(), "ELASTIC/in.modulus.lammps");
		lammps_file(lmp,cfile);

		// Filling the 6x6 Voigt Sitffness tensor with its computed as variables
		// by LAMMPS and conversion from GPa to Pa
		SymmetricTensor<2,2*dim> tmp;
		for(unsigned int k=0;k<2*dim;k++){
			for(unsigned int l=k;l<2*dim;l++)
			{
				char vcoef[1024];
				sprintf(vcoef, "C%d%dall", k+1, l+1);
				tmp[k][l] = *((double *) lammps_extract_variable(lmp,vcoef,NULL))*1.0e+09;
			}
		}

		// Write test... (on the data returned by lammps)

		// Conversion of the 6x6 Voigt Stiffness Tensor into the 3x3x3x3
		// Standard Stiffness Tensor
		for(unsigned int i=0;i<2*dim;i++)
		{
			int k, l;
			if     (i==(3+0)){k=0; l=1;}
			else if(i==(3+1)){k=0; l=2;}
			else if(i==(3+2)){k=1; l=2;}
			else  /*(i<3)*/  {k=i; l=i;}


			for(unsigned int j=0;j<2*dim;j++)
			{
				int m, n;
				if     (j==(3+0)){m=0; n=1;}
				else if(j==(3+1)){m=0; n=2;}
				else if(j==(3+2)){m=1; n=2;}
				else  /*(j<3)*/  {m=j; n=j;}

				loc_rep_stiff[k][l][m][n]=tmp[i][j];
			}
		}

		mdcout << "(MD - init - type " << cellmat << " - repl " << repl << ") "
				<< "Equilibration completed!       " << std::endl;

		// close down LAMMPS
		delete lmp;

	}




	template <int dim>
	void EQMDProblem<dim>::equil (std::string cmat,
							  std::string slocin,
							  std::string qplogloc, std::string scrloc,
							  std::string lengthof, std::string stressof, std::string stiffof, std::string systof,
							  unsigned int rep, double mdts, double mdtem, unsigned int mdnss,
							  unsigned int mdnse, double mdss, double mdsa, std::string mdff)
	{
		cellmat = cmat;

		statelocin = slocin;
		qpreplogloc = qplogloc;
		scriptsloc = scrloc;

		lengthoutputfile = lengthof;
		stressoutputfile = stressof;
		stiffoutputfile = stiffof;
		systemoutputfile = systof;

		repl = rep;

		md_timestep_length = mdts;
		md_temperature = mdtem;
		md_nsteps_sample = mdnss;
		md_nsteps_equil = mdnse;
		md_strain_rate = mdss;
		md_strain_ampl = mdsa;
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
		lammps_equilibration();

		if(this_md_batch_process == 0)
		{
			write_tensor<dim>(lengthoutputfile.c_str(), loc_rep_length);
			write_tensor<dim>(stressoutputfile.c_str(), loc_rep_stress);
			write_tensor<dim>(stiffoutputfile.c_str(), loc_rep_stiff);
		}
	}
}

#endif
