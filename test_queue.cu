#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/c_Queue.h"

__global__ void test_queue_in_shared_memory(int capacity)
{
    extern __shared__ QueueSlot slots[];
    __shared__ int queueParam[4];
    __shared__ Queue queue;
    __shared__ int fullAlert[1];

    if(threadIdx.x == 0)
    {
        queueParam[CAPACITY] = capacity;
        queueParam[REARIDX] = 0;
        queueParam[FRONTIDX] = 0;
        queueParam[NUMWAITINGTASKS] = 0;
        
        queue.set(slots, queueParam);
        //queue = Queue(slots, queueParam);

        fullAlert[0] = 0;
    }
    __syncthreads();

    //Queue queue(slots, queueParam);
    
    int globalThreadId = blockDim.x*blockIdx.x + threadIdx.x;
    
    Task task;
    task.ray = Ray(
        Point(globalThreadId, globalThreadId, globalThreadId),
        Vector3D(globalThreadId, globalThreadId, globalThreadId),
        false
    );

    QueueSlot slot;
    slot.pixelIndex = threadIdx.x;
    slot.task = task;

    queue.enqueue(slot);
    __syncthreads();

    // Task task0;
    // task0.ray = Ray(
    //     Point(globalThreadId, globalThreadId, globalThreadId),
    //     Vector3D(globalThreadId, globalThreadId, globalThreadId),
    //     false
    // );

    // QueueSlot slot0;
    // slot0.pixelIndex = -1;
    // slot0.task = task0;
    slot.pixelIndex = -10;
    if(queue.enqueue(slot) == QueueStatus::QUEUEISFULL) { atomicAdd(&fullAlert[0], 1); }

    // int* paramThread = queue.param();
    // printf("Global threadId: %d, rearIdx: %d, numWaitingTasks: %d\n", globalThreadId, paramThread[1], paramThread[3]);

    __syncthreads();

    if(globalThreadId >= 0)
    {
        QueueSlot slot_;
        queue.dequeue(slot_);
        Vector3D direction = slot_.task.ray.direction();
        printf("Global threadId: %d, slot.pixelIndex: %d, ray.direction: (%f, %f, %f)\n",
         globalThreadId, slot_.pixelIndex, direction.x(), direction.y(), direction.z());
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        printf("BlockIdx.x: %d, rearIdx: %d, frontIdx: %d, fullAlert[0]: %d\n", blockIdx.x, queue.rearIdx(), queue.frontIdx(), fullAlert[0]);
    }
}


int main(int argc, char** argv)
{
    int blocks = 2;
    int threads = 64;

    test_queue_in_shared_memory<<<blocks, threads, (threads)*sizeof(QueueSlot)>>>(threads);
    cudaDeviceSynchronize();
}