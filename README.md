# ECL-MST v1.0

ECL-MST is a fast CUDA implementation for computing a minimum spanning tree (MST) or a minimum spanning forest (MSF) of an undirected graph. It operates on graphs stored in binary CSR format. Converters to this format and several pre-converted graphs can be found at https://cs.txstate.edu/~burtscher/research/ECLgraph/.

The CUDA code consists of the source files ECL-MST_10.cu and ECLgraph.h. The paper referenced below explains the ECL-MST algorithm. Note that ECL-MST is protected by the 3-clause BSD license.

The code can be compiled as follows:

    nvcc -O3 -arch=sm_70 ECL-MST_10.cu -o ecl-mst

To compute the MST of the file graph.egr, enter:

    ./ecl-mst graph.egr


### Publication

A. Fallin, A. Gonzalez, J. Seo, and Martin Burtscher. "A High-Performance MST Implementation for GPUs." Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023. [pdf]

*This work has been supported in part by the National Science Foundation under Award Number 1955367 and by an equipment donation from NVIDIA Corporation.*
