/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

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

// Specifically built header files
#include "headers/read_write.h"
#include "headers/stmd_problem.h"

#include "md_sim.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>

#include <libmuscle/libmuscle.hpp>
#include <ymmsl/ymmsl.hpp>

using libmuscle::Data;
using libmuscle::Instance;
using libmuscle::Message;
using ymmsl::Operator;


/** A simple diffusion model on a 1d grid.
 *
 * The state of this model is a 1D grid of concentrations. It sends out the
 * state on each timestep on 'state_out', and can receive an updated state
 * on 'state_in' at each state update.
 */
void molecular(int argc, char * argv[]) {
	// What are S and O_F here? Is the micro instance shut down after MD simulation (ie at each FE iteration)?
	//
    Instance instance(argc, argv, {
            {Operator::O_I, {"micro_strain_in"}},
            {Operator::S, {"state_in"}},
            {Operator::O_F, {"micro_stress_out"}}});

    while (instance.reuse_instance()) {
        // F_INIT
    	// Receives an MDSim class object and a boolean to know if Hooke's law
    	// should be used instead of MD
        //double t_max = instance.get_setting_as<double>("t_max");
    	//auto msg = instance.receive_with_settings("macro_strains_out", started);
        MDSim<dim> md_simulation = instance.get_setting_as<MDSim<dim>>("md_sim");
        bool approx_md_with_hookes_law = instance.get_setting_as<bool>("hooke");

		STMDProblem<3> stmd_problem (md_batch_communicator, md_batch_pcolor);
		stmd_problem.strain(md_simulation, approx_md_with_hookes_law);

        // O_F
		// Sends back the updated MDSim class object (md_simulation) now containing the stress tensor
        //auto data = Data::grid(U.data(), {U.size()}, {"x"});
        //instance.send("final_state_out", Message(t_cur, data));
		instance.send("micro_stress_out", Message(md_simulation));
        std::cerr << "All done" << std::endl;
    }
}


int main(int argc, char * argv[]) {
	molecular(argc, argv);
    return EXIT_SUCCESS;
}


}
