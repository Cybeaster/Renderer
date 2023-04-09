#include "Utils.hpp"

#include "Math.hpp"
#include "Threads/Thread.hpp"

namespace RAPI
{
uint32 OAsyncUtils::GetDesirableNumOfThreads(uint32 MinPerThread, uint32 Len)
{
	uint16 maxThreads = (Len + MinPerThread - 1) / MinPerThread;

	uint16 hardwareThreads = OThread::hardware_concurrency();

	SMath::Min((hardwareThreads != 0 ? hardwareThreads : 2), maxThreads);
}

} // namespace RAPI
