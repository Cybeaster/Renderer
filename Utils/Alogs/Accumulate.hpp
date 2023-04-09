#pragma once

#include "SimpleThreadPool/ThreadPool.hpp"
#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "TypeTraits.hpp"
#include "Vector.hpp"

#include <numeric>

namespace RAPI
{
template<typename Iterator, typename T>
struct SAccumulateBlock
{
	T operator()(Iterator First, Iterator Last)
	{
		return std::accumulate(First, Last, T());
	}
};

template<typename Iterator, typename T>
T CstmAsyncAccumulate(Iterator First, Iterator Last, T Init)
{
	const uint32 lenght = std::distance(First, Last);

	if (lenght == 0)
	{
		return Init;
	}

	uint32 blockSize = 25;
	uint32 numBlocks = (lenght + blockSize - 1) / blockSize;
	OVector<OFuture<T>> futures(numBlocks - 1);
	OSimpleThreadPool pool;

	Iterator blockStart = First;
	for (uint32 i = 0; i < (numBlocks - 1); ++i)
	{
		Iterator blockEnd = blockStart;

		std::advance(blockEnd, blockSize);

		futures[i] = pool.Submit([=]
		                         { SAccumulateBlock<Iterator, T>(blockStart, blockEnd); });

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

} // namespace RAPI
