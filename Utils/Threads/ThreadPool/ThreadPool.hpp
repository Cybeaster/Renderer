#pragma once
#include "../JoiningThread.hpp"
#include "Functor/Functor.hpp"
#include "Hash.hpp"
#include "Pair.hpp"
#include "Queue.hpp"
#include "Set.hpp"
#include "Thread.hpp"
#include "Types.hpp"
#include "Vector.hpp"

#include <ThreadSafeQueue.hpp>
#include <functional>

namespace RenderAPI
{
namespace Thread
{
using namespace RenderAPI;

struct TTaskID
{
	TTaskID() = default;
	TTaskID(const int64 IDArg) noexcept
	    : ID(IDArg) {}
	TTaskID(const TTaskID& TaskID) noexcept
	    : ID(TaskID.ID) {}

	bool operator>(const TTaskID& Arg) noexcept { return ID > Arg.ID; }
	bool operator<(const TTaskID& Arg) noexcept { return ID < Arg.ID; }
	bool operator==(const TTaskID& Arg) noexcept { return ID == Arg.ID; }

	bool operator>=(const TTaskID& Arg) noexcept { return ID >= Arg.ID; }
	bool operator<=(const TTaskID& Arg) noexcept { return ID <= Arg.ID; }
	bool operator!=(const TTaskID& Arg) noexcept { return ID != Arg.ID; }

	friend bool operator<(const TTaskID& FirstID, const TTaskID& SecondID)
	{
		return FirstID.ID < SecondID.ID;
	}

	friend bool operator>(const TTaskID& FirstID, const TTaskID& SecondID)
	{
		return FirstID.ID > SecondID.ID;
	}

private:
	int64 ID;
};

class TThreadPool
{
	using TCallableInterface = OFunctorBase::TCallableInterface;
	using ThreadQueueElem = TTPair<TCallableInterface*, TTaskID>;

public:
	~TThreadPool();
	TThreadPool(uint32 NumOfThreads);
	TThreadPool(/* args */) = default;

	TTaskID AddTask(TCallableInterface* Function);

	void CreateThread(JoiningThread&& Thread);
	void Wait(const TTaskID& ID);
	void WaitAll();
	bool IsDone(const TTaskID& ID);
	void WaitAndShutDown();

private:
	void Run();
	TTQueue<ThreadQueueElem> TaskQueue;
	TTSet<TTaskID> CompletedTasksIDs;

	TConditionVariable QueueCV;
	TConditionVariable CompletedTaskIdsCV;

	TMutex QueueMutex;
	TMutex CompletedTaskMutex;

	OVector<JoiningThread> Threads;

	TTAtomic<bool> Quite = false;
	TTAtomic<int64> LastID{ 0 };
};
} // namespace Thread

} // namespace RenderAPI
