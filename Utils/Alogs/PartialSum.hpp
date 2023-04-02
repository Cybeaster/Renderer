#pragma once

#include "Logging/Log.hpp"
#include "Threads/Barrier.hpp"
#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "UnitTests/TestGroup.hpp"
#include "Utils.hpp"
#include "Vector.hpp"

#include <numeric>

namespace RenderAPI
{

template<typename Iterator>
void AsyncParallelPartSum(Iterator First, Iterator Last)
{
	using ValueType = typename Iterator::value_type;

	struct SProcessChunk
	{
		void operator()(Iterator Begin, Iterator End, OFuture<ValueType>* PreviousEndValue, OPromise<ValueType>* EndValue)
		{
			try
			{
				Iterator end = End;
				++end;

				std::partial_sum(Begin, End, Begin);
				if (PreviousEndValue)
				{
					ValueType addEnd = PreviousEndValue->get();
					*End += addEnd;
					if (EndValue)
					{
						EndValue->set_value(*End);
					}

					std::for_each(Begin, End, [addEnd](ValueType& Item)
					              {
						              Item += addEnd;
						              RAPI_LOG(Log,
						                       "Added item: {}, End Value: {}, Thread ID : {} ",
						                       TO_STRING(Item), TO_STRING(addEnd),OThreadUtils::GetFormattedThreadID()
						                       ) });
				}
				else if (EndValue)
				{
					EndValue->set_value(*End);
				}
			}
			catch (...)
			{
				if (EndValue)
				{
					EndValue->set_exception(std::current_exception());
				}
				else
				{
					throw;
				}
			}
		}
	};

	auto len = std::distance(First, Last);
	if (len == 0)
	{
		return;
	}

	auto numThreads = OAsyncUtils::GetDesirableNumOfThreads(25, len);

	auto blockSize = len / numThreads;

	OVector<OThread> threads(numThreads - 1);
	OVector<OPromise<ValueType>> endValues(numThreads - 1);

	OVector<OFuture<ValueType>> previousEndValues;
	previousEndValues.reserve(numThreads - 1);
	OJoinThreads joiner(threads);

	Iterator blockStart = First;

	for (uint32 it = 0; it < numThreads; ++it)
	{
		Iterator blockLast = blockStart;
		std::advance(blockLast, blockSize - 1);

		RAPI_LOG(Log, "Processing block from {} to {}", TO_STRING(*blockStart), TO_STRING(*blockLast));

		threads[it] = OThread(SProcessChunk(),
		                      blockStart,
		                      blockLast,
		                      (it != 0) ? &previousEndValues[it - 1] : 0,
		                      &endValues(it));

		blockStart = blockLast;
		++blockStart;

		previousEndValues.push_back(endValues[it].get_future());
	}

	Iterator finalElem = blockStart;

	std::advance(finalElem, std::distance(blockStart, Last) - 1);

	SProcessChunk()(blockStart, finalElem, (numThreads > 1) ? &previousEndValues.back() : 0, 0);
}

template<typename Iterator>
void AsyncPartBufferedSum(Iterator First, Iterator Last)
{
	using ValType = typename Iterator::value_type;

	struct SProcessElement
	{
		void operator()(Iterator First, Iterator Last, OVector<ValType>& Buffer, uint32 It, OBarrier Barrier)
		{
			ValType& ithElem = *(First + It);
			bool updateSource = false;

			for (uint32 step = 0, stride = 1; stride < It; ++step, stride *= 2)
			{
				const ValType& source = (step % 2) ? Buffer[It] : ithElem; // 2

				ValType& dest = (step % 2) ? ithElem : Buffer[It];

				const ValType& addEnd = (step % 2) ? Buffer[It - stride] : *(First + It - stride); // 3

				dest = source + addEnd; // 4

				updateSource = !(step % 2);
				Barrier.Wait(); // 5
			}
			if (updateSource) // 6
			{
				ithElem = Buffer[It];
			}
			Barrier.DoneWaiting(); // 7
		}
	};

	const uint32 len = std::distance(First, Last);

	if (len <= 1)
	{
		return;
	}

	OVector<ValType> buffer(len);
	OBarrier barrier(len);
	OVector<OThread> threads(len - 1);
	OJoinThreads joiner(threads);

	Iterator blockStart = First;
	for (uint32 it = 0; it < len - 1; ++it)
	{
		threads[it] = OThread(SProcessElement(), First, Last, std::ref(buffer), it, std::ref(barrier));
	}

	SProcessElement()(First, Last, buffer, len - 1, barrier);
}

MAKE_TEST(PartialSum)
void SPartialSumTest::Run()
{
	auto fillVec = [](OVector<int32>& vector)
	{
		for (int i = 0; i < 100; ++i)
		{
			vector.push_back(i);
		}
	};

	OVector<int32> testVec;
	fillVec(testVec);
}

} // namespace RenderAPI
