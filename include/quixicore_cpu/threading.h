#pragma once

namespace quixicore_cpu {

// Total parallelism (including the calling thread) used by threaded
// kernels. Default 1: fully synchronous, no worker threads exist. Values
// below 1 clamp to 1.
//
// Kernel results are bit-identical across thread counts: work is
// partitioned by row, never split within a reduction.
void set_num_threads(int n);
int num_threads();

}  // namespace quixicore_cpu
