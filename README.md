# ECL-MST v1.0

ECL-MST is a fast CUDA implementation for computing a minimum spanning tree (MST) or a minimum spanning forest (MSF) of an undirected graph. It operates on graphs stored in binary CSR format. Converters to this format and several pre-converted graphs can be found at https://cs.txstate.edu/~burtscher/research/ECLgraph/.

The CUDA code consists of the source files ECL-MST_10.cu and ECLgraph.h. The paper referenced below explains the ECL-MST algorithm. Note that ECL-MST is protected by the 3-clause BSD license.

The code can be compiled as follows:

    nvcc -O3 -arch=sm_70 ECL-MST_10.cu -o ecl-mst

To compute the MST of the file graph.egr, enter:

    ./ecl-mst graph.egr


### Publication

Alex Fallin, Andres Gonzalez, Jarim Seo, and Martin Burtscher. "A High-Performance MST Implementation for GPUs." Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis. Article 77, pp. 1-13. November 2023. DOI: https://dl.acm.org/doi/10.1145/3581784.3607093

Slides: https://cs.txstate.edu/~burtscher/research/ECL-MST/ECL-MST.pptx

Talk: https://sc23.conference-program.com/presentation/?id=pap489&sess=sess156


**Summary**: ECL-MST is an efficient parallelization of Kruskal's Minimum Spanning Tree algorithm for GPUs. Its edge-centric, data-driven implementation avoids the sorting of edges based on the observation that the minimum element of a list is the same regardless of whether the list is sorted or not. Our implementation demonstrates how a massive parallelization of Kruskal's algorithm converges to that of Boruvka's algorithm, which allows multiple edges to be concurrently included in the MST. The judicious use of atomic operations makes ECL-MST lock-free and fast. Among other optimizations, implicit path compression in the union-find data structure, primarily edge-based processing, and a hybrid parallelization between thread and warp computation allow ECL-MST to achieve large performance gains over state-of-the-art MST codes for GPUs.


*This work has been supported in part by the National Science Foundation under Award Number 1955367 and by an equipment donation from NVIDIA Corporation.*
