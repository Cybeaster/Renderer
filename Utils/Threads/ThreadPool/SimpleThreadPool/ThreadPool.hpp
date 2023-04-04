#pragma once

#include "Functor/Functor.hpp"
#include "Queue.hpp"
#include "ThreadSafeQueue.hpp"
#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "Vector.hpp"
#include "WorkStealingQueue.hpp"
namespace RenderAPI
{

class OSimpleThreadPool
{
public:
	template<typename FuncType>
	OFuture<std::invoke_result<FuncType()>::type> Submit(FuncType Function);

	using TFunctor = OFunctor<void()>;
	using TWorkStealingQueue = OWorkStealingQueue<TFunctor>;

	OSimpleThreadPool();

	~OSimpleThreadPool()
	{
		IsDone = true;
	}

	void RunPendingTask();

private:
	void WorkerThread(uint32 Index);
	bool PopTaskFromLocalQueue(TFunctor& Task);
	bool PopTaskFromPoolQueue(TFunctor& Task);
	bool PopTaskFromOtherThreadQueue(TFunctor& Task);

	using TLocalQueue = OQueue<TFunctor>;

	OVector<OUniquePtr<TWorkStealingQueue>> Queues;
	OThreadSafeQueue<TFunctor> PoolWorkQueue;

	thread_local TWorkStealingQueue* LocalWorkQueue;
	thread_local uint32 LocalIndex;

	OVector<OThread> WorkerThreads;
	OAtomic<bool> IsDone;
	OJoinThreads Joiner{ WorkerThreads };
};

} // namespace RenderAPI
