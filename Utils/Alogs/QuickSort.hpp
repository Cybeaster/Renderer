//
// Created by Cybea on 3/8/2023.
//

#ifndef RENDERAPI_QUICKSORT_HPP
#define RENDERAPI_QUICKSORT_HPP

#include "List.hpp"
#include "ThreadPool.hpp"
#include "Time.hpp"
#include "Utils/Types/Threads/Thread.hpp"

#include <algorithm>

namespace RAPI
{

namespace Algo
{

template<typename Container>
Container SequentialQuickSort(Container Input);

template<typename T>
OList<T> SequentialQuickSort(OList<T> Input)
{
	if (Input.empty())
	{
		return Input;
	}
	OList<T> output;

	output.splice(output.begin(), Input, Input.end());

	T const& pivot = *output.begin();

	auto dividePoint = std::partition(Input.begin(), Input.end(), [&](T const& Value)
	                                  { return Value < pivot; });

	OList<T> lowerPart;
	lowerPart.splice(lowerPart.end(), Input, Input.begin(), dividePoint);

	auto newLower(SequentialQuickSort(Move(lowerPart)));
	auto newHigher(SequentialQuickSort(Move(Input)));

	output.splice(output.end(), newHigher);
	output.splice(output.begin(), newLower);

	return output;
}

template<typename T>
struct SQuickSorter
{
	OThreadPool ThreadPool;

	OList<T> DoSort(OList<T>& Chunk)
	{
		if (Chunk.empty())
		{
			return Chunk;
		}

		OList<T> result;

		result.splice(result.begin(), Chunk, Chunk.begin());

		const T& partitionVal = *result.begin();

		typename OList<T>::iterator dividePoint = std::partition(Chunk.begin(), Chunk.end(), [&](const T& Val)
		                                                         { Val < partitionVal; });
		OList<T> newLowerChunk;

		newLowerChunk.splice(newLowerChunk.end(), Chunk, Chunk.begin(), dividePoint);

		OFuture<OList<T>> newLower = ThreadPool.Submit(std::bind(&SQuickSorter::DoSort, this, Move(newLowerChunk)));

		OList<T> newHigher(DoSort(Chunk));
		result.splice(result.end(), newHigher);

		while (newLower.wait_for(SSeconds(0)) == EFutureStatus::TimeOut)
		{
			ThreadPool.RunPendingTask();
		}
		result.splice(result.begin(), newLower.get());
		return result;
	}
};

template<typename T>
OList<T> AsyncQuickSort(OList<T> Input)
{
	if (Input.empty())
	{
		return Input;
	}

	SQuickSorter<T> sorter;
	sorter.DoSort(Input);
}

} // namespace Algo
} // namespace RAPI
#endif // RENDERAPI_QUICKSORT_HPP
