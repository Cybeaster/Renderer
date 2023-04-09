//
// Created by Cybea on 3/22/2023.
//

#ifndef RENDERAPI_SORTER_HPP
#define RENDERAPI_SORTER_HPP

#include "List.hpp"
#include "Stack/Stack.hpp"
#include "Threads/Thread.hpp"
#include "Time.hpp"
#include "Vector.hpp"
namespace RAPI
{

template<typename T>
class OSorter
{
	struct SChunk
	{
		OList<T> Data;
		OPromise<OList<T>> Promise;
	};

public:
	~OSorter()
	{
		IsEndOfData = true;
		for (auto& thread : Threads)
		{
			thread.join();
		}
	}

	void TrySortChunk();
	OList<T> Sort(OList<T>& Data);
	void SortChunk(const OSharedPtr<SChunk>& Chunk);
	void SortThread();

private:
	OThreadSafeStack<SChunk> Chunks;
	OVector<OThread> Threads;
	const uint32 MaxThreadsCount{ std::thread::hardware_concurrency() - 1 };
	OAtomic<bool> IsEndOfData{ false };
};

template<typename T>
void OSorter<T>::SortChunk(const OSharedPtr<SChunk>& Chunk)
{
	Chunk->Promise.set_value(Sort(Chunk->Data));
}

template<typename T>
void OSorter<T>::SortThread()
{
	while (!IsEndOfData)
	{
		TrySortChunk();
		NThisThread::Yield();
	}
}

template<typename T>
OList<T> OSorter<T>::Sort(OList<T>& Data)
{
	if (Data.empty())
	{
		return Data;
	}
	OList<T> result;

	result.splice(result.begin(), Data.begin());

	const T& partitionValue = *result.begin();

	auto dividePoint = std::partition(Data.begin(), Data.end(), [&](const T& Value)
	                                  {
										Value < partitionValue;
		                                  ; });

	SChunk newLowerChunk;
	newLowerChunk.Data.splice(newLowerChunk.Data.end(), Data, Data.begin(), Data.end(), dividePoint);

	OFuture<OList<T>> newLower = newLowerChunk.Promise.get_future();
	Chunks.Push(Move(newLowerChunk));

	if (Threads.size() < MaxThreadsCount)
	{
		Threads.push_back(OThread(&OSorter<T>::SortThread, this));
	}

	OList<T> newHigher(Sort(Data));
	result.splice(result.end(), newHigher);

	while (newLower.wait_for(SSeconds(0)) != EFutureStatus::Ready)
	{
		TrySortChunk();
	}
	result.splice(result.begin(), newLower.get());
	return result;
}

template<typename T>
void OSorter<T>::TrySortChunk()
{
	OSharedPtr<SChunk> chunk = Chunks.pop();
	if (chunk)
	{
		SortChunk(chunk);
	}
}

template<typename T>
OList<T> ParallelQuickSort(const OList<T>& Input)
{
	if (Input.empty())
	{
		return Input;
	}
	OSorter<T> sorter;
	return sorter.Sort(Input);
}

} // namespace RAPI

#endif // RENDERAPI_SORTER_HPP
