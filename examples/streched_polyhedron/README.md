<!--
* Continue description of the inputs.json file in Description
* Continue the description of outputs, logs, and maybe also the restart system
* Take time to redirect the macro logs to outputs, because it doesn't make sense to store the ouput data in logs
-->
# Streching a macroscopic polyhedron with a Silicon crystal nanostructure

Once you have managed to compile SCEMa, and its dependencies (Deal.II, LAMMPS), you can have a go at this examples which aims at pulling on a few cubic centimeters of silicon.

## Description

A macroscopic polyhedron of 3x3x8cm<sup>3</sup> of material is subject to uniaxial stretching (z-axis) during one microsecond. 

Most of the description of the testing setup and configuration is provided in the file `./inputs.json`, the rest is unfortunately hardcoded (at the moment).
  
The `continuum mesh` block contains the finite element mesh information. The finite element degree and quadrature formula refer directly to parameters of deal.ii which set the spatial interpolation (linear interpolation and two quadrature points per dimension). The mesh is set to contain only one element in the x and y dimension, and two in the longest z dimension.
```
  "continuum mesh":{
    "fe degree": 1,
    "quadrature formula": 2,
    "input": {
      "style" : "cuboid",
      "x length" : 0.03,
      "y length" : 0.03,
      "z length" : 0.08,
      "x cells" : 1,
      "y cells" : 1,
      "z cells" : 2
     }
  }
  ```
  
The whole test is set to last two timesteps, each lasting 0.5 microsecond.
```
  "continuum time":{
    "timestep length": 5.0e-7,
    "start timestep": 1,
    "end timestep": 2
  }
```
  
Finally, for the execution to run smoothly a few paths have to be set in `./inputs.json`.

The paths to the input, output, restart and log files of the finite element (macroscale) and molecular dynamics (nanoscale) simulations:
```
"directory structure":{
    "macroscale input": "./macroscale_input",
    "nanoscale input": "./nanoscale_input",
    "macroscale output": "./macroscale_output",
    "nanoscale output": "./nanoscale_output",
    "macroscale restart": "./macroscale_restart",
    "nanoscale restart": "./nanoscale_restart",
    "macroscale log": "./macroscale_log",
    "nanoscale log": "./nanoscale_log"
  }
```

The path to the execution scripts as well as force field parameters of LAMMPS, used to simulate the nanoscale systems:
```  
"molecular dynamics parameters":{
    "scripts directory": "./lammps_scripts_sisw",
  }
```

Finally, the path to the spline-based algorithm used to filter redundant molecular dynamics simulations (not used in this example):
```  
"model precision":{
    "spline":{
      "scripts directory": "/path/to/SCEMa/spline"
    }
  }
```

## Execution

Except for the workflow's executable that you have previously build (for example at `/path/to/SCEMa/build/`), all necessary files for the execution of the example are provided in `/path/to/SCEMa/examples/stretched_polyhedron/`. This directory can be placed anywhere on the system, once you have chosen its location simply move to it:
```
cd /chosen/path/stretched_polyhedron/
```

Verify one last time that the paths sepcified in `inputs.json` are all set correctly. Let's assume that you have decided not to modify the location of the example directory, you are now at `/path/to/SCEMa/examples/streched_polyhedron/`.

Regarding the input/ouput/restart/log directories, only the input directories need to exist when the simulation is started. Currently, `./nanoscale_input` exists (data regarding molecular structures is stored in it), but `./mcroscale_input` doesn't. Simply create the directory:
```
mkdir ./macroscale_input
```
In the `./nanoscale_input` directory is found:
```
ls nanoscale_input/
> init.sic_1.bin		init.sic_1.length	init.sic_1.stiff	init.sic_1.stress	sic_1.json
```
The files are named as follows `init.{material-name}_{replica-number}.{bin,length, stiff,stress,json}`. They respectively contain for the molecular system `sic_1`: a binary dump of the initial position of atoms, the initial dimensions, the initial homogenised stiffness tensor, the initial stress tensor, and the initial density.

The example can now be executed (for example using 2 processes) using:
```
mpirun -np 2 ../../build/dealammps ./inputs.json
```

The total execution should take approximately four minutes. 

## Analysis

The simplest post-treatment that can be done is to visualise the evolution of the displacements, forces, strains and stresses on the finite element mesh using [ParaView](https://www.paraview.org/). 

The data is located in `./macroscale_log/` stored in the standard VTK file format.

You'll find there two groups of files `history*` and `solution*`, they respectively contain the data of the finite element mesh concerning the quadrature points (stress and strains) and concerning the nodes (forces and displacements).

Files are named as followed `{history,solution}-{timestep}.{process#}.vtu`, hence the multiple files. Conveniently, simply opening the `{history,solution}.pvd` files, enables to access all the data at once.

Here is typically, the displacement field at the end of the simulation (timestep #2), the maximum local displacement is 0.32mm.

<img src="/examples/streched_polyhedron/displacement_field.jpeg" class="full-width">

