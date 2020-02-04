It is assumed here that DeaLAMMPS has been pulled here to a given directory named /path/to/DeaLAMMPS

1. In the directory `/path/to/DeaLAMMPS`, create a build directory:

```
mkdir build
```

2. Move the CMakeLists.txt suited for the machine to the build directory and rename it `CMakeLists.txt` (Note that the  `archer.CMakeLists.text` is the only one maintained): 

```
cp CMakeLists/machine.CMakeLists.txt build/CMakeLists.txt
```

3. Move to the build directory, and generate the MakeFile: 

```
cd build
cmake ./
```

6. Compile the executables: 

```
make
````

   or compile them separately with:

```
make dealammps
```

```
make init_material
```
