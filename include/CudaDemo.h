#ifndef CUDADEMO
#define CUDADEMO

#include <stdio.h>

class CudaDemo
{
private:
   /* data */
public:
   CudaDemo(/* args */);
   ~CudaDemo();
   void SetGPU();
   void Hello();
   void MatrixSum1D();
   void GetGPUProperties(int device_id = 0);
   int GetSPcores(int device_id = 0);
   void GPU_Indicator_Query(int device_id = 0);
   void Grid1d_bLOCK1D();
   void Grid2d_bLOCK1D();
   void Grid2d_bLOCK2D();

};
#endif