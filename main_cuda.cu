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

  int num_bytes, num_spheres, num_planes, num_lights;
  timeval t_start, t_end;
  double elapsed_time;
  Sphere *d_spheres;
  Plane *d_planes;
  Light *d_lights;
	std::string filename = "out.ppm";

  Image* d_image;
  num_bytes = (width * height) * sizeof(Color);
  checkCudaError(cudaMallocManaged(&d_image, num_bytes));
  memset(d_image, 0, num_bytes);

  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks;

	

  gettimeofday (&t_start, NULL);

  // h_image = new Image[width * height];

  if (c_initScene (&d_spheres, &num_spheres, 
        &d_planes, &num_planes,
        &d_lights, &num_lights))
  {
    // //Allocation of memory for the scene on device
    // cudaMalloc (&d_image, num_bytes);

    numBlocks = dim3 (width/threadsPerBlock.x + 1, height/threadsPerBlock.y + 1);

    float tanFov = tan (fov * 0.5 * M_PI / 180.0f);
    float aspect_ratio = float (width) / float (height);

    printf ("Rendering scene:\n");
    printf ("Width: %d \nHeight: %d\nFov: %.2f\n", width, height, fov);

    numBlocks = dim3 (width/threadsPerBlock.x + 1, height/threadsPerBlock.y + 1);

    printf ("Blocks: %d x %d\n", numBlocks.x, numBlocks.y);
  
    float portion = 0.1f;    // the portion of tasks will be put into the shared memory
    int numRay = 40;    // # of rays for antialiasing
    int capacity = portion*numRay*threadsPerBlock.x*threadsPerBlock.y;
    printf("The size of allocated shared memory (Bytes): %lu\n", capacity*sizeof(QueueSlot));
    printf("The size of QueueSlot (Bytes): %lu\n", sizeof(QueueSlot));

    float *d_state;
    cudaMalloc(&d_state, numRay);
    init_stuff<<<1, numRay>>>(time(0), d_state);
    cudaDeviceSynchronize();
    cudaCheckErrors ("Calling kernel k_test");

    k_trace <<<numBlocks, threadsPerBlock, capacity*sizeof(QueueSlot)>>>
    (d_image, d_planes, num_planes, d_spheres, num_spheres, d_lights, 
     num_lights, aspect_ratio, tanFov, width, height, d_state,
     numRay, portion, capacity);
    cudaDeviceSynchronize();
    cudaCheckErrors ("Calling kernel k_test");

    gettimeofday (&t_end, NULL);

    elapsed_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsed_time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    printf ("\r100.00%%");
    printf ("\nFinished!\n");
    printf ("Rendering time: %.3f s\n", elapsed_time/1000.0);
    
    printf("(HOST) d_image color: (%f, %f, %f)\n", d_image[1].r(), d_image[1].g(), d_image[1].b());
    //cudaMemcpy (h_image, d_image, num_bytes, cudaMemcpyDeviceToHost);
    writePPMFile (d_image, "cuda.ppm", width, height);
  }
  else
    printf ("ERROR. Exiting...\n");

  // delete h_image;
  checkCudaError(cudaFree(d_image));
  cudaFree (d_planes);
  cudaFree (d_spheres);
  cudaFree (d_lights);
  
	return 0;
}
