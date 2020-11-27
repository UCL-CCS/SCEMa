#!/bin/bash

if [ -z "$MUSCLE3_HOME" ] ; then
    echo 'Error: MUSCLE3_HOME is not set.'
    echo "Use 'MUSCLE3_HOME=/path/to/muscle3 $0' to run the example"
    exit 1
fi

echo 'Running multiscale molecular continuum in C++'

. python/build/venv/bin/activate
muscle_manager molecular_continuum.ymmsl &

export LD_LIBRARY_PATH=$MUSCLE3_HOME/lib:$LD_LIBRARY_PATH

BINDIR=cpp/build

$BINDIR/molecular --muscle-instance=micro[0] >'micro[0].log' 2>&1 &
$BINDIR/molecular --muscle-instance=micro[1] >'micro[1].log' 2>&1 &
$BINDIR/molecular --muscle-instance=micro[2] >'micro[2].log' 2>&1 &
$BINDIR/load_balancer --muscle-instance=rr >rr.log 2>&1 &
$BINDIR/continuum --muscle-instance=macro >macro.log 2>&1 &

wait
