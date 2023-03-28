#pragma once

#include "Threads/Thread.hpp"
#include "TypeTraits.hpp"
#include "Vector.hpp"

#include <numeric>
template<typename Iterator, typename T>
struct SAccumulateBlock
{
	void operator()(Iterator First, Iterator Last, T& Result)
	{
		Result = std::accumulate(First, Last, Result);
	}
};

template<typename Iterator, typename T>
T AsyncAccumulate(Iterator First, Iterator Last, T Init)
{
	const uint32 lenght = std::distance(First, Last);

	if (lenght == 0)
	{
		return Init;
	}

	uint32 minPerThread = 25;
	uint32 maxThreads = (lenght + minPerThread - 1) / minPerThread;

	uint32 hardwareThreads = OThread::hardware_concurrency();

	uint32 numThreads = std::min(hardwareThreads != 0 ? hardwareThreads : 2, maxThreads);

	uint32 blockSize = lenght / numThreads;

	OVector<T> results(numThreads);

	OVector<OThread> threads(numThreads - 1);

	Iterator blockStart = First;

	for (uint32 i = 0; i < (numThreads - 1); ++i)
	{
		Iterator blockEnd = blockStart;

		std::advance(blockEnd, blockSize);

		threads[i] = std::thread(
		    SAccumulateBlock<Iterator, T>(),
		    blockStart,
		    blockEnd,
		    std::ref(results[i]));
	}

	SAccumulateBlock<Iterator, T>()(blockStart, Last, results[numThreads - 1]);

	std::for_each(threads.begin(),threads.end(),
	    std::mem_fn(&std::thread::join));

	return std::accumulate(results.begin(), results.end(), Init);
}