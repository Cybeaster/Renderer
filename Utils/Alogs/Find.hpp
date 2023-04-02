#pragma once

#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "TypeTraits.hpp"
#include "Vector.hpp"
template<typename Iterator, typename MatchType>
Iterator CstmAsyncFind(Iterator First, Iterator Last, MatchType Type)
{
	struct SFindElem
	{
		void operator()(Iterator Begin, Iterator End, MatchType Match, OPromise<Iterator>* Result, OAtomic<bool>* IsDone)
		{
			try
			{
				for (; (Begin != End) && !IsDone->load(); ++Begin)
				{
					if (*Begin == Match)
					{
						Result->set_value(Begin);
						IsDone->store(true);
						return;
					}
				}
			}
			catch (...)
			{
				try
				{
					Result->set_exception(std::current_exception());
					IsDone->store(true);
				}
				catch (...)
				{
				}
			}
		}
	};

	auto len = std::distance(First, Last);

	if (len == 0)
	{
		return Last;
	}

	uint32 minPerThread = 25;

	uint32 maxThreads = (len + minPerThread - 1) / minPerThread;
	uint32 hrdwrThreads = OThread::hardware_concurrency();
	uint32 numThreads = std::min(hrdwrThreads == 0 ? 2 : hrdwrThreads, maxThreads);

	uint32 blockSize = len / numThreads;

	OPromise<Iterator> result;
	OAtomic<bool> isDone(false);

	OVector<OThread> threads(numThreads - 1);
	{
		OJoinThreads joiner(threads);
		Iterator blockStart = First;
		for (uint32 it = 0; it < numThreads - 1; it++)
		{
			Iterator blockEnd = blockStart;
			std::advance(blockEnd, blockSize);

			threads[it] = OThread(SFindElem(), blockStart, blockEnd, Type, &result, &isDone);
			blockStart = blockEnd;
		}

		SFindElem()(blockStart, Last, Type, &result, &isDone);
	}

	if (!isDone.load())
	{
		return Last;
	}
	return result.get_future().get();
}

namespace
{
template<typename Iterator, typename MatchType>
Iterator AsyncFindImpl(Iterator First, Iterator Last, MatchType Type, OAtomic<bool>& IsDone)
{
	try
	{
		uint32 len = std::distance(First, Last);
		uint32 numPerThread = 25;
		if (len < (2 * numPerThread))
		{
			for (; (First != Last) && !IsDone.load(); ++First)
			{
				if (*First == Type)
				{
					IsDone = true;
					return First;
				}
			}
			return Last;
		}

			Iterator const midPoint = First + (len / 2);
			OFuture<Iterator> result = std::async(&AsyncFindImpl<Iterator, MatchType>, midPoint, Last, Type, std::ref(IsDone));
			auto dirResult = AsyncFindImpl(First, midPoint, Type, IsDone);
			return (dirResult == midPoint) ? result.get() : dirResult;

	}
	catch (...)
	{
		IsDone = true;
		throw;
	}
}
} // namespace

template<typename Iterator, typename MatchType>
Iterator AsyncFind(Iterator Begin, Iterator End, MatchType Match)
{
	OAtomic<bool> isDone(false);
	return AsyncFindImpl(Begin, End, Match, isDone);
}