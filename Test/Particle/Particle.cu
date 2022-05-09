// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "glm.hpp"
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>


__global__ void calcParticlePos(float* particlePos, float* inc, float* result,const float incMultiplier)
{
    int it = threadIdx.x;
    float incResult  = (particlePos[it] - inc[it]) * incMultiplier;
    result[it] = particlePos[it] + 1.f * (incResult - particlePos[it]);

}

__global__ void incVelocity(float* currentVec, float* inc, float* result)
{
    int it = threadIdx.x;
    result[it] = currentVec[it] + inc[it];
}

__global__ void rotate(float* current,float* inc, float* result)
{
    int it = threadIdx.x;
    result[it] = current[it] + inc[it];
}


void rotate(const glm::vec3& particlePos,const glm::vec3& inc,float incMultiplier)
{
    calcParticlePos<<<>>>;
}