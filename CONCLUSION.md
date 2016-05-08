# Project Conclusions
By Jason Yao

## CUDA Server
The following data was compiled and tested on **CUDA server 1**.

## How block and grid sizes/dimensions were chosen
Since all of the CUDA compute servers on CIMS runs off new architecture
with a `Compute Capability` of `2.x -> 3.x`, that means that (from a hardware perspective) 
each block is limited to **1,024** threads per block.

Another reason to stick with 1,024 threads per block was because it was a nice
multiple of the warp size (which is currently 32 for these machines).

From this, the block dimensions are simply **32 x 32**. For the grid dimensions, it
was decided to just use a square formation. From the data's `n` value, it will thus be:

- `ceil(sqrt(n)/32) x ceil(sqrt(n)/32)`

## Compilation code
```sh
nvcc -g -m64 \
--generate-code arch=compute_30,code=sm_30 \
--generate-code arch=compute_35,code=sm_35 \
--generate-code arch=compute_37,code=sm_37 \
--generate-code arch=compute_50,code=sm_50 \
--generate-code arch=compute_52,code=sm_52 \
--generate-code arch=compute_52,code=compute_52 \
-o maxgpu maxgpu.cu
```

## The graph





## The graphs
### Speedup
The speedup due to thread count is shown in the graph below:
- For `N = 1,000`

![The speedup due to thread count](img/speedupA.png)

- For `N = 10,000`

![The speedup due to thread count](img/speedupB.png)

- For both speedup graphs combined

![The speedup due to thread count](img/speedup.png)

Where we define speedup per Dr.Zahran's notes as:

![Process speedup definition](img/speedupDefinition.png)

This speedup graph was derived from the averages of the time required to execute fully, 
shown in the graph below, and taken from a sample size of 5 runs per each set, 
with the full data available for viewing [here](Aggregate Output.xlsx).

### Runtime
The original wall-clock runtimes are shown in the following graphs:
- For `N = 1,000`

![Program runtime](img/runtimeA.png)

- For `N = 10,000`

![Program runtime](img/runtimeB.png)

- For both runtime graphs combined

![Program runtime](img/runtime.png)

### Performance
We define the performance of an executing program per Dr. Zahran's definition as:

![Performance Definition](img/performanceDefinition.png)

From this definition, the following performance graphs were derived from the full data set linked above:

- For `N = 1,000`

![Thread performance](img/performanceA.png)

- For `N = 10,000`

![Thread performance](img/performanceB.png)

- For both performance graphs combined

![Thread performance](img/performance.png)

## The "Why" for the speedup graphs
As we can see per the trendline in the speedup graphs, there is an exponential decrease in 
speedup as the number of threads increases. This could be attributed to the fact that as the
number of threads increase, the performance cost of `fork()`ing and `join()`ing threads increases
as well, thus making the overhead increase as the number of threads increases.

Another thing to note from the speedup graphs is that as the value of `N` increases, generally 
speaking there is a noticeable decrease in the speedup. This could be attributed to the fact that
since each thread has more data to work on, it requires more time to process during each parallel
section.
