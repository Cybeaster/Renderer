#pragma once
#include "TypeTraits.hpp"

#include <execution>
namespace RAPI
{

namespace NAsyncAlgoPolices
{
using Sequenced = std::execution::sequenced_policy;
using Parallel = std::execution::parallel_policy;
using ParallelUnsequenced = std::execution::parallel_unsequenced_policy;
} // namespace NAsyncAlgoPolices

class OAsyncUtils
{
public:
	static uint32 GetDesirableNumOfThreads(uint32 MinPerThread, uint32 Len);
};

} // namespace RAPI
