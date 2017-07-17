#!/bin/bash
for CELL in {0..100}
do
  for QP in {0..7}
  do
    echo ${CELL}-${QP}
    cp $(ls -t *.${CELL}-${QP}.stiff | head -1) last.${CELL}-${QP}.stiff
  done
done
