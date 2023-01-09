#include "../SmartPtr.hpp"
#include "../Thread.hpp"
#include "../Queue.hpp"
#pragma once

namespace RenderAPI
{
    template <typename T>
    class TThreadSafeQueue
    {

    public:
        TThreadSafeQueue(const TThreadSafeQueue &Queue)
        {
            TMutexGuard guard(Mutex);
            InternalQueue = Queue.InternalQueue;
        }
        ~TThreadSafeQueue();

        void Push(T Value);

        void Pop(T &Value);
        TTSharedPtr<T> Pop();

        bool Empty();

    private:
        mutable TMutex Mutex;
        TTQueue<T> InternalQueue;
        TConditionVariable PushCondition;
    };

    template <typename T>
    void TThreadSafeQueue<T>::Push(T Value)
    {
        TMutexGuard guard(Mutex);
        InternalQueue.push(Value);
        PushCondition.notify_one();
    }

    template <typename T>
    void TThreadSafeQueue<T>::Pop(T &Value)
    {

        TUniqueLock lock(Mutex);
        PushCondition.wait(lock, [this]()
                           { return !InternalQueue.empty(); });

        Value = InternalQueue.front();
        InternalQueue.pop();
    }

    template <typename T>
    TTSharedPtr<T> TThreadSafeQueue<T>::Pop()
    {
        TUniqueLock lock(Mutex);
        PushCondition.wait(lock, [this]()
                           { return !InternalQueue.empty(); });
        TTSharedPtr<T> result(MakeShared(InternalQueue.front()));
        InternalQueue.pop();
        return result;
    }

    template <typename T>
    bool TThreadSafeQueue<T>::Empty()
    {
        TMutexGuard guard(Mutex);
        return InternalQueue.empty();
    }

} // namespace RenderAPI
