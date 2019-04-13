#include <stdio.h>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>

extern "C" { 
#include "cuda_link.h" 
    }


 __global__ void gpu_scattering ( int i, int N, int total_sc, double flu_part, double *f0, int *atyp, int jj, double qvect1, double qvect2, double qvect3, double (*cq)[3], double (*CELL)[3], double scellx, double scelly, double scellz, cuDoubleComplex *dev_dummy_array_1, cuDoubleComplex *dev_dummy_array_2 )
 {
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    double test_exp;        
    test_exp = qvect1 * (cq[ii][0]  + CELL[i][0] * scellx) + qvect2 * (cq[ii][1]  + CELL[i][1] * scelly) + qvect3 * (cq[ii][2]  + CELL[i][2] * scellz)  ; 
    dev_dummy_array_1[ii] = make_cuDoubleComplex( 0.0 , test_exp );
 }

extern "C"
int cuda_function( int i ,
                    int N ,
                    int total_sc ,
                    double flu_part ,
                    int lcount ,
                    int rcount,
                    double *f0,
                    int *atyp,
                    int jj,
                    double qvect1,
                    double qvect2,
                    double qvect3,
                    double (*cq)[3],
                    double (*CELL)[3],
                    double scellx,
                    double scelly,
                    double scellz,
                    double *dummy_array_1,
                    double  *dummy_array_2 )
   {
    
   int  *dev_atyp;
   double *dev_f0;
   double *dev_cq[3];
   double *dev_CELL[3];
   cuDoubleComplex *dev_dummy_array_1;
   cuDoubleComplex *dev_dummy_array_2;
    

    cudaMalloc( (void**)& dev_f0 , rcount * lcount * sizeof (double) );
    cudaMalloc( (void**)& dev_CELL , total_sc * 3 * sizeof (double) );
    cudaMalloc( (void**)& dev_cq , N * 3 * sizeof (double) );
    cudaMalloc( (void**)& dev_atyp ,N * sizeof (int) );
    cudaMalloc( (void**)& dev_dummy_array_1 , N * sizeof(cuDoubleComplex)  ); 
    cudaMalloc( (void**)& dev_dummy_array_2 , N * sizeof(cuDoubleComplex)  ); 
   
   
    cudaMemcpyAsync( dev_f0, f0, rcount * lcount * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dev_CELL, CELL , total_sc * 3 * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dev_cq, cq , N * 3 * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dev_atyp, atyp , N * sizeof(int), cudaMemcpyHostToDevice );
//    cudaMemcpyAsync( dev_dummy_array_1, dummy_array_1, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );
  //  cudaMemcpyAsync( dev_dummy_array_2, dummy_array_2, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice );

 gpu_scattering<<<1, 10>>>( i, N, total_sc, flu_part, dev_f0, dev_atyp, jj, 
                              qvect1,  qvect2,  qvect3, cq, CELL, scellx, 
                              scelly, scellz, dev_dummy_array_1, dev_dummy_array_2 ) ;//best: constant number of blocks + threads (64+128)

   return 0;
    
    }
