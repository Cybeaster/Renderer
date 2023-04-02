#pragma once

#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "TypeTraits.hpp"
#include "Vector.hpp"

#include <numeric>
template<typename Iterator, typename T>
struct SAccumulateBlock
{
	T operator()(Iterator First, Iterator Last)
	{
		return std::accumulate(First, Last, T());
	}
};

namespace
{

template<typename Iterator, typename T>
T CstmAsyncAccumulate(Iterator First, Iterator Last, T Init)
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

	OVector<OFuture<T>> futures(numThreads - 1);
	OVector<OThread> threads(numThreads - 1);

	OJoinThreads joinThreads(threads);

	Iterator blockStart = First;
	for (uint32 i = 0; i < (numThreads - 1); ++i)
	{
		Iterator blockEnd = blockStart;

		std::advance(blockEnd, blockSize);

		OPackagedTask<T(Iterator, Iterator)> task{ SAccumulateBlock<Iterator, T>() };

		futures[i] = task.get_future();

		threads[i] = OThread(Move(task), blockStart, blockEnd);
		blockStart = blockEnd;
	}

	T lastResult = SAccumulateBlock<Iterator, T>()(blockStart, Last);

	T result = Init;

	for (auto& future : futures)
	{
		result += future.get();
	}
	result += lastResult;
	return result;
}
} // namespace

template<typename Iterator, typename T>
T AsyncAccumulate(Iterator First, Iterator Last, T Init)
{
	const auto lenght = std::distance(First, Last);

	uint32 maxChunkSize = 25;
	if (lenght <= maxChunkSize)
	{
		return std::accumulate(Init);
	}

	Iterator mid = First;

	std::advance(mid, lenght / 2);
	OFuture<T> firstHalfRes = std::async(AsyncAccumulate<Iterator, T>, First, mid, Init);

	T secondHalfRes = AsyncAccumulate(mid, Last, T());

	return firstHalfRes.get() + secondHalfRes;
}
