#pragma once
#include "../JoiningThread.hpp"
#include "Functor/Functor.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Queue.hpp"
#include "Thread.hpp"
#include "Pair.hpp"
#include "Hash.hpp"
#include <functional>
#include "Set.hpp"
namespace RenderAPI
{
    namespace Thread
    {
        using namespace RenderAPI;

        struct TTaskID
        {
            TTaskID() = default;
            TTaskID(const int64 IDArg) noexcept : ID(IDArg) {}
            TTaskID(const TTaskID &TaskID) noexcept : ID(TaskID.ID) {}

            bool operator>(const TTaskID &Arg) noexcept { return ID > Arg.ID; }
            bool operator<(const TTaskID &Arg) noexcept { return ID < Arg.ID; }
            bool operator==(const TTaskID &Arg) noexcept { return ID == Arg.ID; }

            bool operator>=(const TTaskID &Arg) noexcept { return ID >= Arg.ID; }
            bool operator<=(const TTaskID &Arg) noexcept { return ID <= Arg.ID; }
            bool operator!=(const TTaskID &Arg) noexcept { return ID != Arg.ID; }

            friend bool operator<(const TTaskID &FirstID, const TTaskID &SecondID)
            {
                return FirstID.ID < SecondID.ID;
            }

            friend bool operator>(const TTaskID &FirstID, const TTaskID &SecondID)
            {
                return FirstID.ID > SecondID.ID;
            }

        private:
            int64 ID;
        };

        class TThreadPool
        {
            using TCallableInterface = TFunctorBase::TCallableInterface;
            using ThreadQueueElem = TTPair<TFunctorBase::TCallableInterface *, TTaskID>;

        public:
            TThreadPool(/* args */) = delete;

            ~TThreadPool();
            TThreadPool(uint32 NumOfThreads);

            TTaskID AddTask(TCallableInterface *Function);

            void CreateThread(JoiningThread &&Thread);
            void Wait(const TTaskID &ID);
            void WaitAll();
            bool IsDone(const TTaskID &ID);
            void WaitAndShutDown();

        private:
            void Run();
            TTQueue<ThreadQueueElem> TaskQueue;
            TTSet<TTaskID> CompletedTasksIDs;

            TConditionVariable QueueCV;
            TConditionVariable CompletedTaskIdsCV;

            TMutex QueueMutex;
            TMutex CompletedTaskMutex;

            TTVector<JoiningThread> Threads;

            TTAtomic<bool> Quite = false;
            TTAtomic<int64> LastID{0};
        };
    } // namespace Thread

} // namespace RenderAPI
