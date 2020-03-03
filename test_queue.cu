#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/c_Queue.h"

__global__ void test_queue_in_shared_memory(int capacity)
{
    extern __shared__ QueueSlot slots[];
    __shared__ int queueParam[4];
    
    if(threadIdx.x == 0)
    {
        queueParam[CAPACITY] = capacity;
        queueParam[REARIDX] = REARIDX;
        queueParam[FRONTIDX] = FRONTIDX;
        queueParam[NUMWAITINGTASKS] = 0;
    }
    __syncthreads();

    Queue queue(slots, queueParam);
    
    int globalThreadId = blockDim.x*blockIdx.x + threadIdx.x;
    
    Task task;
    task.ray = Ray(Point(globalThreadId, globalThreadId, globalThreadId), Vector3D(1, 1, 1));

    QueueSlot slot;
    slot.pixelIndex = globalThreadId;
    slot.task = task;

    queue.enqueue(slot);

    int* paramThread = queue.param();
    //printf("Global threadId: %d, rearIdx: %d, numWaitingTasks: %d\n", globalThreadId, paramThread[1], paramThread[3]);

}


int main(int argc, char** argv)
{

    int blocks = 1;
    int threads = 33;

    test_queue_in_shared_memory<<<blocks, threads, (threads + 1)*sizeof(QueueSlot)>>>(threads + 1);
    cudaDeviceSynchronize();
}