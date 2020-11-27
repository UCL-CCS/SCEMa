#include <cstdlib>
#include <cinttypes>
#include <random>

#include <libmuscle/libmuscle.hpp>
#include <ymmsl/ymmsl.hpp>


using libmuscle::Data;
using libmuscle::Instance;
using libmuscle::Message;
using ymmsl::Operator;
using ymmsl::Settings;

#include "headers/stmd_sync.h"

/* A proxy which divides many calls over few instances.
 *
 * Put this component between a driver and a set of models, or between a
 * macro model and a set of micro models. It will let the driver or macro-
 * model submit as many calls as it wants, and divide them over the available
 * (micro)model instances in a round-robin fashion.
 *
 * Assumes a fixed number of micro-model instances.
 */
void load_balancer(int argc, char * argv[]) {

    mmd_problem = new STMDSync<dim> (mmd_communicator, mmd_pcolor);

    mmd_problem->init(start_timestep, md_timestep_length, md_temperature,
					   md_nsteps_sample, md_strain_rate, md_force_field, nanostatelocin,
					   nanostatelocout, nanostatelocres, nanologloc,
					   nanologloctmp, nanologlochom, macrostatelocout,
					   md_scripts_directory, freq_checkpoint, freq_output_homog,
					   machine_ppn, mdtype, cg_dir, nrepl,
					   use_pjm_scheduler, input_config, approx_md_with_hookes_law);

    // Make sure here that the load balancer instance sends a message to the macro instance that
    // the init() has completed correctly because the macro instance needs the replicas data (for the
    // stiffness matrix

    Instance instance(argc, argv, {
            {Operator::F_INIT, {"macro_strains_out[]"}},
            {Operator::O_I, {"micro_strain_in[]"}},
            {Operator::S, {"micro_stress_out[]"}},
            {Operator::O_F, {"macro_stresses_in[]"}}});

    while (instance.reuse_instance(false)) {
        // F_INIT
        int started = 0;
        int done = 0;

        // Receives the ScaleBridgingData and prepare the MDSims
        // Equivalent to std::vector< MDSim<dim> > STMDSync<dim>::prepare_md_simulations(ScaleBridgingData scale_bridging_data)

        // what is num_calls? (number of QPs or number ot timesteps?)
        int num_calls = instance.get_port_length("macro_strains_out");

        instance.set_port_length("macro_stresses_in", num_calls);

        // what is num_workers? (number of MDSim? number of CPUs?)
        int num_workers = instance.get_port_length("micro_strain_in");

        // Split MDSims to send them individually to each micro instance
        // mmd_problem->update() subfunctions need to be made public
        //mmd_problem->update(timestep, present_time, newtonstep, scale_bridging_data);

        while (done < num_calls) {
            // This loop should be equivalent to
            // STMDSync<dim>::execute_inside_md_simulations(std::vector<MDSim<dim> >& md_simulations)
            while ((started - done < num_workers) && (started < num_calls)) {
                auto msg = instance.receive_with_settings("macro_strains_out", started);
                instance.send("micro_strain_in", msg, started % num_workers);
                ++started;
            }
            // This is void STMDSync<dim>::share_stresses(std::vector<MDSim<dim> >& md_simulations)
            // but above all void STMDSync<dim>::store_md_simulations(std::vector<MDSim<dim> > md_simulations,
    		// ScaleBridgingData& scale_bridging_data)
            auto msg = instance.receive_with_settings("micro_stress_out", done % num_workers);
            instance.send("macro_stresses_in", msg, done);
            ++done;
        }
    }
}


int main(int argc, char * argv[]) {
    load_balancer(argc, argv);
    return EXIT_SUCCESS;
}

