#!/bin/bash

# Simulation directory
sim_folder=$1

# Target directory location
save_folder=$2

# Make the save folder if it doesnt exist
mkdir -p $save_folder

# Save dealammps configuration
cp $sim_folder/bak.dealammps.cc $save_folder/

# Save log file
cp $sim_folder/log.dealammps $save_folder/

# Create archives
tar -czvf $save_folder/macrostate_in.tar.gz $sim_folder/macroscale_state/in > $save_folder/tmp.min &
tar -czvf $save_folder/nanostate_in.tar.gz $sim_folder/nanoscale_state/in > $save_folder/tmp.nin  &
tar -czvf $save_folder/macrostate_restart.tar.gz $sim_folder/macroscale_state/restart > $save_folder/tmp.mre  &
tar -czvf $save_folder/nanostate_restart.tar.gz $sim_folder/nanoscale_state/restart > $save_folder/tmp.nre &
tar -czvf $save_folder/macrolog.tar.gz $sim_folder/macroscale_log > $save_folder/tmp.mlo  &
tar -czvf $save_folder/nanolog_spec.tar.gz $sim_folder/nanoscale_log/spec > $save_folder/tmp.nlo &
