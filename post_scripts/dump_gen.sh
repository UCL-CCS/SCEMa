#!/bin/bash
lammps_exec=/home/maxime/source/lammps-17Nov16/src/lmp_mpi
lammps_script=/home/maxime/local_projects/postDeaLAMMPS/scripts/dump.lammps
data_dir=$1

cd $data_dir
mkdir visualisation

find . -name "*.bin" -printf "%f\n" > tmp.list
while read p
do
  lammps_bin=$p
  mpirun -np 4 $lammps_exec -in $lammps_script -screen none -var locd $lammps_bin < /dev/null
  echo ${lammps_bin}
done < tmp.list
rm tmp.* log.*

vmd -f `ls -l visualisation/*.0.PE_1.bin.xyz | sort -n -k 9 | awk '{print $NF}'`
