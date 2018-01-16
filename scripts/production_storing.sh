#!/bin/bash

# save_folder='attemptX_tsYY-ZZ'
save_folder=$1
# store_dir='/epsrc/e283/e283/vassaux/dealammps/arc_3pbt/'
# store_dir=$2

mkdir $save_folder

cp log.dealammps $save_folder"/"
cp -r macroscale_log $save_folder"/"
cp -r nanoscale_state/out/spec $save_folder"/nano_spec"

cp -r macroscale_state/restart macroscale_state/in/
cp -r nanoscale_state/restart nanoscale_state/in/

tar -czvf $save_folder"/"macro_restart.tar.gz macroscale_state/restart/*
tar -czvf $save_folder"/"nano_restart.tar.gz nanoscale_state/restart/*

# cp -r $save_folder $store_dir
# cp -r macroscale_state/in/restart $store_dir$save_folder/macro_restart
# cp -r nanoscale_state/in/restart $store_dir$save_folder/nano_restart

#make outclean
