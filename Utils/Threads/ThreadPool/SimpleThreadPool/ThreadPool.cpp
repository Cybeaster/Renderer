//
// Created by Cybea on 4/2/2023.
//

#include "ThreadPool.hpp"

#include <memory>

namespace RenderAPI
{

void OSimpleThreadPool::WorkerThread()
{
	LocalWorkQueue = std::make_unique<TLocalQueue>();

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
			WorkerThreads.emplace_back(&OSimpleThreadPool::WorkerThread, this);
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
	if (LocalWorkQueue && !LocalWorkQueue->empty())
	{
		task = Move(LocalWorkQueue->front());
		LocalWorkQueue->pop();
		task();
	}
	else if (WorkQueue.TryPop(task))
	{
		task();
	}
	else
	{
		NThisThread::Yield();
	}
}

template<typename FuncType>
OFuture<std::invoke_result<FuncType()>::type> OSimpleThreadPool::Submit(FuncType Function)
{
	using TResult = typename std::invoke_result<FuncType()>::type;
	OPackagedTask<TResult()> newTask(Move(Function));

	OFuture<TResult> result(newTask.get_future());

	if (LocalWorkQueue)
	{
		LocalWorkQueue->push(Move(newTask));
	}
	else
	{
		WorkQueue.Push(Move(newTask));
	}
	return result;
}

} // namespace RenderAPI