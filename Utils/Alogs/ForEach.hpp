#pragma once

#include "Threads/Utils.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "Vector.hpp"

#include <iterator>
#include <utility>

namespace RAPI
{
namespace
{

template<typename Iterator, typename Func>
void CustAsyncForEach(Iterator First, Iterator Last, Func Function)
{
	const auto len = std::distance(First, Last);

	if (len == 0)
	{
		return;
	}

	auto numThreads = OAsyncUtils::GetDesirableNumOfThreads(25, len);

	auto blockSize = len / numThreads;

	OVector<OFuture<void>> futures(numThreads - 1);
	OVector<OThread> threads(numThreads - 1);
	OJoinThreads joiner(threads);

	Iterator blockStart = First;

	for (uint32 it = 0; it < numThreads - 1; ++it)
	{
		Iterator blockEnd = blockStart;
		{
			std::advance(blockEnd, blockSize);
			OPackagedTask<void(void)> task([=]
			                               { std::for_each(blockStart, blockEnd, Function); });

			futures[it] = task.get_future();
			threads[it] = OThread(Move(task));
		}
		blockStart = blockEnd;
	}

	std::for_each(blockStart, Last, Function);

	for (auto& future : futures)
	{
		future.get();
	}
}
} // namespace

template<typename Iterator, typename Func>
void AsyncForEach(Iterator First, Iterator Last, Func&& Function)
{
	auto len = std::distance(First, Last);

	if (len == 0)
	{
		return;
	}

	uint32 minPerThread = 25;

	if (len < (2 * minPerThread))
	{
		std::for_each(First, Last, Function);
	}
	else
	{
		const Iterator mid = First + len / 2;
		auto firstHalf = std::async(&AsyncForEach<Iterator, Func>, First, mid, Forward(Function));

		AsyncForEach(mid, Last, Function);
		firstHalf.get();
	}
}
} // namespace RAPI