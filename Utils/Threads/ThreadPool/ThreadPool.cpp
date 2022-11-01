#include "ThreadPool.hpp"

namespace RenderAPI
{

    namespace Thread
    {
        TThreadPool::TThreadPool(uint32 NumOfThreads)
        {
            Threads.reserve(NumOfThreads);
            for (size_t i = 0; i < NumOfThreads; i++)
            {
                Threads.emplace_back(&TThreadPool::Run, this);
            }
        }
        
        TTaskID TThreadPool::AddTask(TTFunctor<void()> &&Function)
        {
            int64 taskID = LastID++;
            TMutexGuard queueLock(QueueMutex);
            TTaskID id(taskID);
            auto elem = std::pair(Function, taskID);
            // TaskQueue.emplace(std::move(elem));

            QueueCV.notify_one();
            return TTaskID(taskID);
        }

        void TThreadPool::Run()
        {
            while (!Quite)
            {
                TUniqueLock uniqueLock(QueueMutex);
                QueueCV.wait(uniqueLock, [this]() -> bool
                             { return !TaskQueue.empty() || Quite; });

                if (!TaskQueue.empty())
                {
                    auto &elem = std::move(TaskQueue.front());
                    TaskQueue.pop();
                    uniqueLock.unlock();
                    // elem.first();

                    TMutexGuard guardlock(CompletedTaskMutex);
                    CompletedTasksIDs.insert(elem.second);
                    CompletedTaskIdsCV.notify_all();
                }
            }
        }
        void TThreadPool::Wait(const TTaskID &ID)
        {
            TTaskID id{23};
            TTaskID id1{0};

            TUniqueLock lock(CompletedTaskMutex);
            // wait for notify in function run
            CompletedTaskIdsCV.wait(lock, [this, ID]() -> bool
                                    { return CompletedTasksIDs.find(ID) != CompletedTasksIDs.end(); });
        }
        void TThreadPool::WaitAll()
        {
            TUniqueLock lock(QueueMutex);
            CompletedTaskIdsCV.wait(lock, [this]() -> bool
                                    {
                                        TMutexGuard taskLock(CompletedTaskMutex);
                                        return  TaskQueue.empty() && LastID == CompletedTasksIDs.size(); });
        }

        TThreadPool::~TThreadPool()
        {
            Quite = true;
            for (auto &&thread : Threads)
            {
                QueueCV.notify_all();
                thread.Join();
            }
        }

        bool TThreadPool::IsDone(const TTaskID &ID)
        {
            TMutexGuard lock(CompletedTaskMutex);
            if (CompletedTasksIDs.find(ID) != CompletedTasksIDs.end())
            {
                return true;
            }
            return false;
        }

        void TThreadPool::CreateThread(JoiningThread &&Thread)
        {
            Threads.push_back(std::move(Thread));
        }
    } // namespace Thread

} // namespace RenderAPI
