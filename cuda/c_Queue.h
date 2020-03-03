#ifndef QUEUE_H_
#define QUEUE_H_

#include "c_QueueSlot.h"

enum class QueueStatus
{
    QUEUEISWORKING, QUEUEISFULL, QUEUEISEMPTY, NONE
};

enum QueueParam
{
    CAPACITY = 0,
    REARIDX,
    FRONTIDX,
    NUMWAITINGTASKS
};

class Queue
{
public:

    __host__ __device__ Queue(QueueSlot* slots, int* queueParam) 
    {
        _slots = slots;
        _param = queueParam;
    }

    __device__ inline QueueSlot operator[](int index) const { return _slots[index]; }
    __device__ inline QueueSlot &operator[](int index) { return _slots[index]; }

    __device__ inline int capacity() const { return _param[CAPACITY]; }
    __device__ inline int rearIdx() const { return _param[REARIDX]; }
    __device__ inline int frontIdx() const { return _param[FRONTIDX]; }
    __device__ inline int length() const { return _param[NUMWAITINGTASKS]; }
    __device__ inline int* param() const { return _param; }

    __device__ inline int handleNumWaitingTasks(int sign)
    {
        return atomicAdd(&_param[NUMWAITINGTASKS], sign);
    }

    __device__ inline bool isFull(int currentNumWaitingTasks)
    {
        if (currentNumWaitingTasks >= _param[CAPACITY])
        {
            atomicExch(&_param[NUMWAITINGTASKS], _param[CAPACITY]);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool isFull()
    {
        if (handleNumWaitingTasks(1) >= _param[CAPACITY])
        {
            atomicExch(&_param[NUMWAITINGTASKS], _param[CAPACITY]);
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
            atomicExch(&_param[NUMWAITINGTASKS], 0);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool isEmpty()
    {
        if (handleNumWaitingTasks(1) <= 0)
        {
            atomicExch(&_param[NUMWAITINGTASKS], 0);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ inline bool isReachEndIndex(int index)
    {
        return (index == _param[CAPACITY])? true : false;
    }

    __device__ int handleIndex(int sign)
    {
        int resultIdx;

        // ring-like index of the queue
        if (sign == 1) // if we want to increase _rearIdx
        {
            resultIdx = atomicCAS(&_param[REARIDX], _param[CAPACITY], 0);

            // if _rearIdx reaches the end of the queue, put it back to the beginning of the queue
            if (isReachEndIndex(resultIdx))
            {
                return 0;
            }
        
            resultIdx = atomicAdd(&_param[REARIDX], 1);
            
        }
        else if (sign == -1)
        {
            resultIdx = atomicCAS(&_param[FRONTIDX], _param[CAPACITY], 0);

            // if _rearIdx reaches the beginning of the queue, put it back to the end of the queue
            if (isReachEndIndex(resultIdx))
            {  
                return 0;
            }

            resultIdx = atomicAdd(&_param[FRONTIDX], 1);
        }

        return resultIdx;
    }

    __device__ QueueStatus enqueue(QueueSlot &data)
    {
        int currentNumWaitingTasks = handleNumWaitingTasks(1);
        //printf("Number of waiting tasks: %d. From global threadId: %d\n", currentNumWaitingTasks, threadIdx.x + blockIdx.x*blockDim.x);

        QueueStatus status = QueueStatus::NONE;

        if (!isFull(currentNumWaitingTasks)) // if queue is not full
        {
            int index = handleIndex(1);

            // store data in the queue
            _slots[index] = data;

            //printf("Store in queue[%d], local threadId: %d, global threadId: %d\n", index, queue[index], threadIdx.x + blockIdx.x*blockDim.x);

            status = QueueStatus::QUEUEISWORKING;
            return status;
        }
        else
        {
            //printf("\nQueue is full. From global threadId %d. _rearIdx: %d. # of waiting tasks%d\n", threadIdx.x + blockIdx.x*blockDim.x, _rearIdx, _numWaitingTasks);
            status = QueueStatus::QUEUEISFULL;
            return status;
        }
    }

    __device__ QueueStatus dequeue(QueueSlot &data)
    {
        int currentNumWaitingTasks = handleNumWaitingTasks(-1);
        //printf("Number of waiting tasks: %d. From global threadId %d\n", currentNumWaitingTasks, threadIdx.x + blockIdx.x*blockDim.x);
        
        QueueStatus status = QueueStatus::NONE;

        if (!isEmpty(currentNumWaitingTasks)) // if queue have waiting tasks
        {
            int index = handleIndex(-1);
            
            // get data from the queue
            data = _slots[index];

            //printf("Get from queue[%d], local threadId: %d, global threadId: %d\n", index, queue[index], threadIdx.x + blockIdx.x*blockDim.x);

            status = QueueStatus::QUEUEISWORKING; 
            return status;
        }
        else
        {
            //printf("\nQueue is empty. From global threadId %d. _frontIdx: %d. # of Waiting Tasks: %d\n", threadIdx.x + blockIdx.x*blockDim.x, _frontIdx, _numWaitingTasks);
            status = QueueStatus::QUEUEISEMPTY;
            return status;
        }
    }

private:
    QueueSlot* _slots;
    int* _param;
};

#endif