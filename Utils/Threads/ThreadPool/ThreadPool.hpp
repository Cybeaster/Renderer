#pragma once
#include "../JoiningThread.hpp"
#include "Functor/Functor.hpp"
#include "Pair.hpp"
#include "Queue.hpp"
#include "Set.hpp"
#include "ThreadSafeQueue.hpp"
#include "Types.hpp"
#include "Utils/Types/HashMap/Hash.hpp"
#include "Utils/Types/Threads/Thread.hpp"
#include "Vector.hpp"

#include <functional>

namespace RenderAPI::Async
{
struct STaskID
{
	STaskID() = default;
	explicit STaskID(const int64 IDArg) noexcept
	    : ID(IDArg) {}

	STaskID(const STaskID& TaskID) = default;

	bool operator>(const STaskID& Arg) const noexcept { return ID > Arg.ID; }
	bool operator<(const STaskID& Arg) const noexcept { return ID < Arg.ID; }
	bool operator==(const STaskID& Arg) const noexcept { return ID == Arg.ID; }

	bool operator>=(const STaskID& Arg) const noexcept { return ID >= Arg.ID; }
	bool operator<=(const STaskID& Arg) const noexcept { return ID <= Arg.ID; }
	bool operator!=(const STaskID& Arg) const noexcept { return ID != Arg.ID; }

private:
	int64 ID;
};

class OThreadPool
{
	using CallableInterfaceType = SFunctorBase::SICallableInterface;
	using ThreadQueueElemType = OPair<CallableInterfaceType*, STaskID>;

public:
	~OThreadPool();
	explicit OThreadPool(uint32 NumOfThreads);
	OThreadPool(/* args */) = default;

	STaskID AddTask(CallableInterfaceType* Function);

	void CreateThread(JoiningThread&& Thread);
	void Wait(const STaskID& ID);
	void WaitAll();
	bool IsDone(const STaskID& ID);
	void WaitAndShutDown();

private:
	void Run();
	OQueue<ThreadQueueElemType> TaskQueue;
	OSet<STaskID> CompletedTasksIDs;

	OConditionVariable QueueCV;
	OConditionVariable CompletedTaskIdsCV;

	OMutex QueueMutex;
	OMutex CompletedTaskMutex;

	OVector<JoiningThread> Threads;

	OAtomic<bool> Quite = false;
	OAtomic<int64> LastID{ 0 };
};
} // namespace RenderAPI::Async
