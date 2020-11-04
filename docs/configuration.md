Most of the configuration of the multiscale simulation is featured in a JSON file passed as an argument to the `dealammps` executable:
```
mpirun /path/to/SCEMa/build/dealammps configuration.json
```

## Outlook
The JSON configuration file must be shaped as follows:
```
{
  "problem type":{
    "class": "dogbone" or "compact" or "dropweight",
    "strain rate": 0.002
  },
  "scale-bridging":{
    "stress computation method": 0 (molecular model) or 1 (analytical hooke's law) or 2 (surrogate model),
    "approximate md with hookes law": 0 (normal mode) or 1 (debug mode, replaces LAMMPS kernel with simple dot product operation),
    "use pjm scheduler": 0
  },
  "continuum time":{
    "timestep length": 5.0e-7,
    "start timestep": 1,
    "end timestep": 500
  },
  "continuum mesh":{
    "fe degree": 1,
    "quadrature formula": 2,
    "input": {
      "style" : "cuboid" (for dogbone or dropweight) or "file3D" (for dogbone or compact),
      "x length" : 0.03,
      "y length" : 0.03,
      "z length" : 0.08,
      "x cells" : 3,
      "y cells" : 3,
      "z cells" : 8
     }
  },
  "model precision":{
    "md":{
      "min quadrature strain norm": 1.0e-10
    },
    "clustering":{
      "points": 10 (number of points in the spline approximation of the strain trajectory),
      "min steps": 5 (number of steps before the clustering algorithm kicks in, if 5 then algorithm starts at timestep 6), 
      "diff threshold": 0.000001 (when the L2-norm distance of 2 splines exceeds this threshold they are considered different), 
      "scripts directory": "./clustering" (directory where the python scripts for the clustering algorithm are located)
    }
  },
  "molecular dynamics material":{
    "number of replicas": 1,
    "list of materials": ["matname"],
    "distribution": {
      "style": "uniform",
      "proportions": [1.0]
    },
    "rotation common ground vector":[1.0, 0.0, 0.0]
  },
  "molecular dynamics parameters":{
    "temperature": 300.0,
    "timestep length": 2.0,
    "strain rate": 1.0e-4,
    "number of sampling steps": 100,
    "scripts directory": "./lammps_scripts_opls" or "./lammps_scripts_reax",
    "force field": "opls" or "reax"
  },
  "computational resources":{
    "machine cores per node": 24,
    "maximum number of cores for FEM simulation": 10,
    "minimum number of cores for MD simulation": 1
  },
  "output data":{
    "checkpoint frequency": 100,
    "visualisation output frequency": 1,
    "analytics output frequency": 1,
  "loaded boundary force output frequency": 1,
    "homogenization output frequency": 1000
  },
  "directory structure":{
    "macroscale input": "./macroscale_input",
    "nanoscale input": "./nanoscale_input",
    "macroscale output": "./macroscale_output",
    "nanoscale output": "./nanoscale_output",
    "macroscale restart": "./macroscale_restart",
    "nanoscale restart": "./nanoscale_restart",
    "macroscale log": "./macroscale_log",
    "nanoscale log": "./nanoscale_log" or "none" (if no nanoscale log)
  }
}

```

