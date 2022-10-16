#include "../JoiningThread.hpp"
#include "Functor/Functor.hpp"
#include "Types.hpp"
#include "Vector.hpp"
#include "Queue.hpp"
#include "Thread.hpp"
#include "Pair.hpp"
#include "Hash.hpp"
#include "Set.hpp"
namespace RenderAPI
{
    namespace Thread
    {
        using namespace RenderAPI::Functor;

        struct TTaskID
        {
            explicit TTaskID(const int64 IDArg) : ID(IDArg){}
            int64 ID;
        };

        class ThreadPool
        {

        public:
            ThreadPool(/* args */) = delete;

            ~ThreadPool();
            ThreadPool(uint32 NumOfThreads);

            template <typename ReturnType, typename... Args>
            TTaskID AddTask(const TTFunctor<ReturnType(Args...)>& Function, Args&&... Arguments);

            void CreateThread(JoiningThread &&Thread);
            void Wait(const TTaskID &ID);
            void WaitAll();
            void IsDone(const TTaskID &ID);
            void WaitAndShutDown();
            
        private:

            void Run();

            TTQueue<TTPair<TTFunctor<void()>, TTaskID>> TaskQueue;
            TTSet<TTaskID> CompletedTasksIDs;

            TConditionVariable QueueCV;
            TConditionVariable CompletedTaskIdsCV;

            TMutex QueueMutex;
            TMutex CompletedTaskMutex;

            TTVector<JoiningThread> Threads;

            TTAtomic<bool> Quite = false;
            TTAtomic<int64> LastID = 0;
        };

    } // namespace Thread

} // namespace RenderAPI
