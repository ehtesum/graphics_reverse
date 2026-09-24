#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void coop_kernel(int* out, int iters) {
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < iters; ++i) {
        grid.sync();
    }
    if (grid.thread_rank() == 0) {
        *out = 42;
    }
}
