#ifndef MD_SIM_H
#define MD_SIM_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>

#include <deal.II/base/symmetric_tensor.h>

namespace HMM { 
	
	template <int dim>
	class MDSim {
		public:
			MDSim()
			{
			}
	
			int qp_id;
			int replica;
			int material;
		
			std::string time_id; 

			SymmetricTensor<2,dim> strain; //input
			SymmetricTensor<2,dim> stress; //output

			bool				stress_updated = false; 
		
			std::string stress_out_file; // replace with stress tensor 
			std::string output_file;
			std::string restart_file;
			std::string log_file;
			std::string md_scripts_directory;
		
			double 			timestep_length;
			double			temperature;
			int 				nsteps_sample;
			double 			strain_rate;
			std::string force_field;
		
			bool output_homog; // what is this? seems to add an extra dump of atom coords	
			bool checkpoint;

			void define_file_names(std::string nanologloctmp, std::string macrostatelocout)
			{
  	  	stress_out_file = macrostatelocout + "/last." + std::to_string(qp_id) + "." + std::to_string(replica) + ".stress";
    	  log_file = nanologloctmp + "/" + time_id  + "." + std::to_string(qp_id) + "." + std::to_string(material) + "_" + std::to_string(replica);
				// Preparing directory to write MD simulation log files
				mkdir(log_file.c_str(), ACCESSPERMS);
			}		
		private:
	};	

}

#endif 
