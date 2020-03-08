#ifndef QUEUESLOT_H_
#define QUEUESLOT_H_

#include "c_Ray.h"

typedef struct
{
  Ray ray;
  float intensity;
} Task;

struct QueueSlot
{
    int pixelIndex;
    Task task;
};

#endif