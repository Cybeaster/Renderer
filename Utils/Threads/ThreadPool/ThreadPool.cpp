#include "ThreadPool.hpp"

namespace RenderAPI
{

    namespace Thread
    {
        ThreadPool::ThreadPool(uint32 NumOfThreads)
        {
            Threads.reserve(NumOfThreads);
            for (size_t i = 0; i < NumOfThreads; i++)
            {
                Threads.emplace_back(&ThreadPool::Run, this);
            }
        }

        template <typename ReturnType, typename... Args>
        TTaskID ThreadPool::AddTask(const TTFunctor<ReturnType(Args...)>& Function, Args&&... Arguments)
        {
            int64 taskID = LastID++;
            TMutexGuard queueLock(QueueMutex);
            
            TaskQueue.emplace();
        }

        void ThreadPool::Run()
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
                    elem.first();

                    TMutexGuard guardlock(CompletedTaskMutex);
                    CompletedTasksIDs.insert(elem.second);
                    CompletedTaskIdsCV.notify_all();
                }
            }
        }

        ThreadPool::~ThreadPool()
        {
        }

        void ThreadPool::CreateThread(JoiningThread &&Thread)
        {
            Threads.push_back(std::move(Thread));
        }
    } // namespace Thread

} // namespace RenderAPI
