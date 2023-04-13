
#include "NamedThreadPool.hpp"

namespace RAPI
{

ONamedThreadPool::ONamedThreadPool()
{
	InitThreads();
}

void ONamedThreadPool::InitThreads()
{
	for (uint8 it = 0; it < NamedThreadsCount; ++it)
	{
		Threads[static_cast<EThreadID>(it)] = {
			OThread(&ONamedThreadPool::Run, this),
			OUniquePtr<TQueue>(new TQueue())
		};
	}
}

} // namespace RAPI
