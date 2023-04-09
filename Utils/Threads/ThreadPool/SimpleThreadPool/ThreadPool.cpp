//
// Created by Cybea on 4/2/2023.
//

#include "ThreadPool.hpp"

#include <memory>

namespace RAPI
{

void OSimpleThreadPool::WorkerThread(uint32 Index)
{
	LocalIndex = Index;
	LocalWorkQueue = Queues[LocalIndex].get();

	while (!IsDone)
	{
		RunPendingTask();
	}
}
OSimpleThreadPool::OSimpleThreadPool()
{
	const uint32 threadCount = OThread::hardware_concurrency();

	try
	{
		for (uint32 it = 0; it < threadCount; it++)
		{
			Queues.emplace_back(new TWorkStealingQueue);
		}

		for (uint32 it = 0; it < threadCount; it++)
		{
			WorkerThreads.emplace_back(&OSimpleThreadPool::WorkerThread, this, it);
		}
	}
	catch (...)
	{
		IsDone = true;
		throw;
	}
}

void OSimpleThreadPool::RunPendingTask()
{
	TFunctor task;
	if (PopTaskFromLocalQueue(task)
	    || PopTaskFromPoolQueue(task)
	    || PopTaskFromOtherThreadQueue(task))
	{
		task();
	}
	else
	{
		NThisThread::Yield();
	}
}
bool OSimpleThreadPool::PopTaskFromLocalQueue(OSimpleThreadPool::TFunctor& Task)
{
	return (LocalWorkQueue != nullptr) && LocalWorkQueue->TryPop(Task);
}

bool OSimpleThreadPool::PopTaskFromPoolQueue(OSimpleThreadPool::TFunctor& Task)
{
	return PoolWorkQueue.TryPop(Task);
}
bool OSimpleThreadPool::PopTaskFromOtherThreadQueue(OSimpleThreadPool::TFunctor& Task)
{
	for (uint32 it = 0; it < Queues.size(); it++)
	{
		uint32 index = (LocalIndex + it + 1) % Queues.size();
		if (Queues[index]->TrySteal(Task))
		{
			return true;
		}
	}
	return false;
}

template<typename FuncType>
OFuture<std::invoke_result<FuncType()>::type> OSimpleThreadPool::Submit(FuncType Function)
{
	using TResult = typename std::invoke_result<FuncType()>::type;
	OPackagedTask<TResult()> newTask(Move(Function));

	OFuture<TResult> result(newTask.get_future());

	if (LocalWorkQueue)
	{
		LocalWorkQueue->Push(Move(newTask));
	}
	else
	{
		PoolWorkQueue.Push(Move(newTask));
	}
	return result;
}

} // namespace RAPI