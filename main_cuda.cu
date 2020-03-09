/*
 * main_cuda.cu
 *
 *  Created on: Oct 30, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#include "c_util.h"

#include <math.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaError(e) { checkCudaErrorImpl(e, __FILE__, __LINE__); }

inline void checkCudaErrorImpl(cudaError_t e, const char* file, int line, bool abort = true) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s - %s:%d\n", cudaGetErrorString(e), file, line);
        if (abort) exit(e);
    }
}

int main (int argc, char** argv)
{
  if (argc < 3)
	{
		printf ("Usage: %s <widht> <height> [<fov>]\n", argv[0]);
		return 0;
	}

	int width = atoi (argv[1]);;
	int height = atoi (argv[2]);;
  float fov = 60.0;
  
  if (argc >= 4)
  {
    fov = atof (argv[3]);
  }

  timeval t_start, t_end;
  double elapsed_time_w_work_stealing, elapsed_time_wo_work_stealing;
  int num_bytes, num_spheres, num_planes, num_lights;
  Sphere *d_spheres;
  Plane *d_planes;
  Light *d_lights;

  Color blackColor(0.0f);
  Color whiteColor(1.0f);
  Image* d_image;
  Image* record_image;
  num_bytes = (width * height) * sizeof(Color);
  checkCudaError(cudaMallocManaged(&d_image, num_bytes));
  checkCudaError(cudaMallocManaged(&record_image, num_bytes));
  // memset(d_image, blackColor, num_bytes);

  int* record;
  checkCudaError(cudaMallocManaged(&record, width*height*sizeof(int)));

  for(int k = 0; k < width*height; k++)
  {
    d_image[k] = blackColor;
    record_image[k] = blackColor;
    record[k] = 0;
  }

  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks;

  if (c_initScene (&d_spheres, &num_spheres, 
        &d_planes, &num_planes,
        &d_lights, &num_lights))
  {
    // Allocation of memory for the scene on device
    numBlocks = dim3 ((int)ceil((float)width/threadsPerBlock.x), (int)ceil((float)height/threadsPerBlock.y));
    printf ("Blocks: %d x %d\n", numBlocks.x, numBlocks.y);

    float tanFov = tan (fov * 0.5 * M_PI / 180.0f);
    float aspect_ratio = float (width) / float (height);

    printf ("Width: %d \nHeight: %d\nFov: %.2f\n", width, height, fov);

    float portion = 0.4f;    // the portion of tasks will be put into the shared memory
    int numRay = 10;    // # of rays for antialiasing
    int capacity = portion*numRay*threadsPerBlock.x*threadsPerBlock.y;
    printf("The size of allocated shared memory (Bytes): %lu\n", capacity*sizeof(QueueSlot));
    printf("The number of stolen rays : %f\n", numRay*portion);

    float *d_state;
    cudaMalloc(&d_state, numRay);
    init_stuff<<<1, numRay>>>(time(0), d_state);
    cudaDeviceSynchronize();
    cudaCheckErrors ("Calling kernel k_test");

    // w/ work stealing
    gettimeofday (&t_start, NULL);

    k_trace <<<numBlocks, threadsPerBlock, capacity*sizeof(QueueSlot)>>>
    (d_image, d_planes, num_planes, d_spheres, num_spheres, d_lights, 
    num_lights, aspect_ratio, tanFov, width, height, d_state,
    numRay, portion, capacity, record);
    cudaDeviceSynchronize();
    cudaCheckErrors ("Calling kernel k_test");

    gettimeofday (&t_end, NULL);
    elapsed_time_w_work_stealing = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsed_time_w_work_stealing += (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    printf ("(w/) Rendering time: %.3f s\n", elapsed_time_w_work_stealing/1000.0);

    // w/o work stealing
    gettimeofday (&t_start, NULL);

    k_trace_without_work_stealing <<<numBlocks, threadsPerBlock, capacity*sizeof(QueueSlot)>>>
    (d_image, d_planes, num_planes, d_spheres, num_spheres, d_lights, 
    num_lights, aspect_ratio, tanFov, width, height, d_state,
    numRay, portion, capacity, record);
    cudaDeviceSynchronize();
    cudaCheckErrors ("Calling kernel k_test");

    gettimeofday (&t_end, NULL);
    elapsed_time_wo_work_stealing = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsed_time_wo_work_stealing += (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    printf ("(w/o) Rendering time: %.3f s\n", elapsed_time_wo_work_stealing/1000.0);
//     for(int u = 0; u < width*height; u++)
//     {
//       if(record[u] < numRay)
//       {
//         printf("pixelIndex: (%d, %d) only has been stolen %d rays\n", u%width, u/width, record[u]);
//         //Color color_(1.0f/float(record[u]));
//         record_image[u] = whiteColor;
//       }
//     }
    double speedup = 100*(elapsed_time_wo_work_stealing - elapsed_time_w_work_stealing)/elapsed_time_wo_work_stealing;
    printf("\nSpeedup: %f%%\n", speedup);
    printf ("\nAll Finished!\n");
    
    

    writePPMFile(d_image, "cuda.ppm", width, height);
    writePPMFile(record_image, "record.ppm", width, height);

    writePPMFile(d_image, "cuda_without_work_stealing.ppm", width, height);
    writePPMFile(record_image, "record_without_work_stealing.ppm", width, height);
  }
  else
    printf ("ERROR. Exiting...\n");

  // delete h_image;
  checkCudaError(cudaFree(d_image));
  cudaFree(d_planes);
  cudaFree(d_spheres);
  cudaFree(d_lights);
  
	return 0;
}
