#pragma once

#include "Functor/Functor.hpp"
#include "ThreadSafeQueue.hpp"
#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "Vector.hpp"
namespace RenderAPI
{

class OSimpleThreadPool
{
public:
	template<typename FuncType>
	OFuture<std::invoke_result<FuncType()>::type> Submit(FuncType Function);

	using TFunctor = OFunctor<void()>;
	OSimpleThreadPool();

	~OSimpleThreadPool()
	{
		IsDone = true;
	}

private:
	void WorkerThread();

	OThreadSafeQueue<TFunctor> WorkQueue;
	OVector<OThread> WorkerThreads;
	OAtomic<bool> IsDone;
	OJoinThreads Joiner{ WorkerThreads };
};

} // namespace RenderAPI
