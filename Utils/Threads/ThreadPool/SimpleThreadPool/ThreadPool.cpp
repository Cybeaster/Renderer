//
// Created by Cybea on 4/2/2023.
//

#include "ThreadPool.hpp"

namespace RenderAPI
{

void OSimpleThreadPool::WorkerThread()
{
	while (!IsDone)
	{
		TFunctor task;

		if (WorkQueue.TryPop(task))
		{
			task();
		}
		else
		{
			NThisThread::Yield();
		}
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
template<typename FuncType>
OFuture<std::invoke_result<FuncType()>::type> OSimpleThreadPool::Submit(FuncType Function)
{
	using TResult = typename std::invoke_result<FuncType()>::type;
	OPackagedTask<TResult()> newTask(Move(Function));

	OFuture<TResult> result(newTask.get_future());
	WorkQueue.Push(Move(newTask));

	return result;
}

} // namespace RenderAPI