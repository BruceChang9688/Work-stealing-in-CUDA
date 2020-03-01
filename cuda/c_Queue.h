#ifndef QUEUE_H_
#define QUEUE_H_

#include "c_QueueSlot.h"

enum QueueStatus
{
    QUEUEISWORKING, QUEUEISFULL, QUEUEISEMPTY, NONE
}

class Queue
{
public:
    __host__ __device__ Queue() {}
    __host__ __device__ Queue(const int capacity) 
    {
        _capacity = capacity;
        QueueSlot slots[_capacity];
        _slots = slots[0];
    }

    __device__ inline int handleNumWaitingTasks(int sign)
    {
        return atomicAdd(&_numWaitingTasks, sign);
    }

    __device__ inline bool isFull(int currentNumWaitingTasks)
    {
        if (currentNumWaitingTasks >= _capacity)
        {
            atomicExch(&_numWaitingTasks, _capacity);
            return true;
        }
        else
        {
            return false;
        }
    }

    __host__ __device__ inline bool isFull()
    {
        if (handleNumWaitingTasks(1) >= _capacity)
        {
            atomicExch(&_numWaitingTasks, _capacity);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool isEmpty(int currentNumWaitingTasks)
    {
        if (currentNumWaitingTasks <= 0)
        {
            atomicExch(&_numWaitingTasks, 0);
            return true;
        }
        else
        {
            return false;
        }
    }

    __host__ __device__ inline bool isEmpty()
    {
        if (handleNumWaitingTasks(1) <= 0)
        {
            atomicExch(&_numWaitingTasks, 0);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool isReachEndIndex(int index)
    {
        return (index == _capacity)? true : false;
    }

    __device__ int handleIndex(int sign)
    {
        int resultIdx;

        // ring-like index of the queue
        if (sign == 1) // if we want to increase _rearIdx
        {
            resultIdx = atomicCAS(&_rearIdx, _capacity, 0);

            // if _rearIdx reaches the end of the queue, put it back to the beginning of the queue
            if (isReachEndIndex(resultIdx))
            {
                return 0;
            }
        
            resultIdx = atomicAdd(&_rearIdx, 1);
            
        }
        else if (sign == -1)
        {
            resultIdx = atomicCAS(&_frontIdx, _capacity, 0);

            // if _rearIdx reaches the beginning of the queue, put it back to the end of the queue
            if (isReachEndIndex(resultIdx))
            {  
                return 0;
            }

            resultIdx = atomicAdd(&_frontIdx, 1);
        }

        return resultIdx;
    }

    __device__ int enqueue(QueueSlot &data)
    {
        int currentNumWaitingTasks = handleNumWaitingTasks(1);
        //printf("Number of waiting tasks: %d. From global threadId: %d\n", currentNumWaitingTasks, threadIdx.x + blockIdx.x*blockDim.x);

        QueueStatus status = None;

        if (!isFull(currentNumWaitingTasks)) // if queue is not full
        {
            int index = handleIndex(1);

            // store data in the queue
            for (int offset = 0; offset < 1; offset++)
            {
                _slots[index] = data[offset];
            }
            //printf("Store in queue[%d], local threadId: %d, global threadId: %d\n", index, queue[index], threadIdx.x + blockIdx.x*blockDim.x);

            status = QUEUEISWORKING;
            return status;
        }
        else
        {
            //printf("\nQueue is full. From global threadId %d. _rearIdx: %d. # of waiting tasks%d\n", threadIdx.x + blockIdx.x*blockDim.x, _rearIdx, _numWaitingTasks);
            status = QUEUEISFULL;
            return status;
        }
    }

    __device__ int dequeue(QueueSlot &data)
    {
        int currentNumWaitingTasks = handleNumWaitingTasks(-1);
        //printf("Number of waiting tasks: %d. From global threadId %d\n", currentNumWaitingTasks, threadIdx.x + blockIdx.x*blockDim.x);
        
        QueueStatus status = None;

        if (!isEmpty(currentNumWaitingTasks)) // if queue have waiting tasks
        {
            int index = handleIndex(-1);
            
            // get data from the queue
            for (int offset = 0; offset < 1; offset++)
            {
                data[offset] = _slots[index + offset];
            }
            //printf("Get from queue[%d], local threadId: %d, global threadId: %d\n", index, queue[index], threadIdx.x + blockIdx.x*blockDim.x);

            status = QUEUEISWORKING; 
            return status;
        }
        else
        {
            //printf("\nQueue is empty. From global threadId %d. _frontIdx: %d. # of Waiting Tasks: %d\n", threadIdx.x + blockIdx.x*blockDim.x, _frontIdx, _numWaitingTasks);
            status = QUEUEISEMPTY;
            return status;
        }
    }

private:
    QueueSlot *_slots;
    int _capacity;
    int _frontIdx = 0;
    int _rearIdx = 0;
    int _numWaitingTasks = 0;
}

#endif