#include "ThreadPool.hpp"

namespace RenderAPI::Async
{
OThreadPool::OThreadPool(uint32 NumOfThreads)
{
	Threads.reserve(NumOfThreads);
	for (size_t i = 0; i < NumOfThreads; i++)
	{
		Threads.emplace_back(&OThreadPool::Run, this);
	}
}

STaskID OThreadPool::AddTask(CallableInterfaceType* Function)
{
	int64 taskID = LastID++;
	OMutexGuard queueLock(QueueMutex);

	TaskQueue.emplace(std::make_pair(Function, STaskID(taskID)));

	QueueCV.notify_one();
	return STaskID(taskID);
}

void OThreadPool::Run()
{
	while (!Quite)
	{
		OUniqueLock uniqueLock(QueueMutex);
		QueueCV.wait(uniqueLock, [this]() -> bool
		             { return !TaskQueue.empty() || Quite; });

		if (!TaskQueue.empty())
		{
			auto& elem = TaskQueue.front();
			TaskQueue.pop();
			uniqueLock.unlock();
			elem.first->Call();

			OMutexGuard guardlock(CompletedTaskMutex);
			CompletedTasksIDs.insert(elem.second);
			CompletedTaskIdsCV.notify_all();
		}
	}
}
void OThreadPool::Wait(const STaskID& ID)
{
	OUniqueLock lock(CompletedTaskMutex);
	// wait for notify in function run
	CompletedTaskIdsCV.wait(lock, [this, ID]() -> bool
	                        { return CompletedTasksIDs.find(ID) != CompletedTasksIDs.end(); });
}
void OThreadPool::WaitAll()
{
	OUniqueLock lock(QueueMutex);
	CompletedTaskIdsCV.wait(lock, [this]() -> bool
	                        {
                                        OMutexGuard taskLock(CompletedTaskMutex);
                                        return  TaskQueue.empty() && LastID == CompletedTasksIDs.size(); });
}

OThreadPool::~OThreadPool()
{
	Quite = true;
	for (auto&& thread : Threads)
	{
		QueueCV.notify_all();
		thread.Join();
	}
}

bool OThreadPool::IsDone(const STaskID& ID)
{
	OMutexGuard lock(CompletedTaskMutex);
	return CompletedTasksIDs.find(ID) != CompletedTasksIDs.end();
}

void OThreadPool::CreateThread(JoiningThread&& Thread)
{
	Threads.push_back(std::move(Thread));
}
} // namespace RenderAPI::Async
