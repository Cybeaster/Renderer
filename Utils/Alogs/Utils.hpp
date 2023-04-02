#pragma once
#include "TypeTraits.hpp"

namespace RenderAPI
{

class OAsyncUtils
{
public:
	static uint32 GetDesirableNumOfThreads(uint32 MinPerThread, uint32 Len);
};

} // namespace RenderAPI
