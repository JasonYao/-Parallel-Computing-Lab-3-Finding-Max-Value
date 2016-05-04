# Parallel Computing Lab 3: Finding the Max Value
By Jason Yao, [github](https://github.com/JasonYao/Parallel-Computing-Lab-3-Finding-Max-Value)

## FOR THE GRADER: 
This README utilises github's markdown, and is a much easier read on the github website listed above.

Link: https://github.com/JasonYao/Parallel-Computing-Lab-3-Finding-Max-Value

For my conclusion, please see [here](CONCLUSION.md)

## Description
This program is designed to find the maximum value given an integer `n` such that an array of `long`s
of size `n` is created, and filled with randomised integers in the range `[0, n - 1]`.

Both parallel and sequential versions of this code are provided in the source, and can both
are run by default with the attached compile script.

## Compilation & Running
In order to test the parallel version of the code, the sequential version must be run first, that way it can be `diff`'d to find
any differences with the parallel version.

### To compile the code and test against an input automatically
To change the upperbound, simply change `x=5` to whatever `n` value you would like to serve as the upper bound.
```sh
./compileAndTest
```

### To compile and run the code manually
```sh
nvcc -g -o <output_file_name> <source_file_name>
./<output_file_name> <N_value_upper_bound>
```

e.g. To compile and find the maximum with an upper bound of 10
```sh
nvcc -g -o maxgpu maxgpu.cu
./maxgpu 10
```

Where:
- `nvcc` is the name of the CUDA compiler and linker that will compile, 
and link the libraries required for the source code to run.

- `-g` is a flag to produce debugging information

- `-o` is to signify the output binary

## The output
The output will be a statement proclaiming what the found maximum was. TODO add a picture

## Debugging version
If you'd like to run the debugging version of the code irrespective of the other flags, please edit the source file [maxgpu.cu](maxgpu.cu) and change line **9** from

```sh
bool IS_DEBUG_MODE = false;
```
to
```sh
bool IS_DEBUG_MODE = true;
```
then compile again before running

## Sequential version
If you'd like to run the sequential version of the code instead, please edit the source file [gs.c](gs.c) and change line **10** from

```sh
bool IS_SEQUENTIAL_MODE = false;
```
to
```sh
bool IS_SEQUENTIAL_MODE = true;
```
then compile again before running

## Timed version
If you'd like to run the timed version of this code, irrespective of the other flags, please edit the source file [gs.c](gs.c) and change line **11** from

```sh
bool IS_TIMED_MODE = false;
```
to
```sh
bool IS_TIMED_MODE = true;
```
then compile again before running

## License
This repo is licensed under the terms of the GNU GPL v3, a copy of which may be found [here](LICENSE).
