#pragma once

#include "Threads/Thread.hpp"
#include "Types.hpp"

#include <deque>
namespace RAPI
{

template<typename DataType>
class OWorkStealingQueue
{
public:
	OWorkStealingQueue() = default;
	REMOVE_COPY_FOR(OWorkStealingQueue)

	void Push(DataType Data)
	{
		OMutexGuard lock(Mutex);
		Queue.push_front(Move(Data));
	}

	bool Empty() const
	{
		OMutexGuard lock(Mutex);
		return Queue.empty();
	}

	bool TryPop(DataType& Result)
	{
		OMutexGuard lock(Mutex);
		if (Queue.empty())
		{
			return false;
		}

		Result = Move(Queue.front());
		Queue.pop_front();
		return true;
	}

	bool TrySteal(DataType& Result)
	{
		OMutexGuard lock(Mutex);

		if (Queue.empty())
		{
			return false;
		}
		Result = Move(Queue.back());
		Queue.pop_back();
		return true;
	}

private:
	std::deque<DataType> Queue;
	mutable OMutex Mutex;
};
} // namespace RAPI