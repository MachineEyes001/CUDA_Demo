#include "include/CudaDemo.h"

int main(){
    CudaDemo test;
    // test.Hello();
    // test.SetGPU();
    // test.MatrixSum1D();
    // test.GetGPUProperties(0);
    // test.GetSPcores(0);
    // test.Grid1d_bLOCK1D();
    // test.Grid2d_bLOCK1D();
    // test.Grid2d_bLOCK2D();
    test.GPU_Indicator_Query(0);
    return 0;
}