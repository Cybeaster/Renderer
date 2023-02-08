#include "../Queue.hpp"
#include "../SmartPtr.hpp"
#include "../Thread.hpp"

#pragma once

namespace RenderAPI
{
template<typename T>
class OhreadSafeQueue
{
public:
	OhreadSafeQueue(const OhreadSafeQueue& Queue)
	{
		OMutexGuard guard(Mutex);
		InternalQueue = Queue.InternalQueue;
	}
	~OhreadSafeQueue();

	void Push(T Value);

	void Pop(T& Value);
	OSharedPtr<T> Pop();

	bool Empty();

private:
	mutable OMutex Mutex;
	TTQueue<T> InternalQueue;
	OConditionVariable PushCondition;
};

template<typename T>
void OhreadSafeQueue<T>::Push(T Value)
{
	OMutexGuard guard(Mutex);
	InternalQueue.push(Value);
	PushCondition.notify_one();
}

template<typename T>
void OhreadSafeQueue<T>::Pop(T& Value)
{
	OUniqueLock lock(Mutex);
	PushCondition.wait(lock, [this]()
	                   { return !InternalQueue.empty(); });

	Value = InternalQueue.front();
	InternalQueue.pop();
}

template<typename T>
OSharedPtr<T> OhreadSafeQueue<T>::Pop()
{
	OUniqueLock lock(Mutex);
	PushCondition.wait(lock, [this]()
	                   { return !InternalQueue.empty(); });
	OSharedPtr<T> result(MakeShared(InternalQueue.front()));
	InternalQueue.pop();
	return result;
}

template<typename T>
bool OhreadSafeQueue<T>::Empty()
{
	OMutexGuard guard(Mutex);
	return InternalQueue.empty();
}

} // namespace RenderAPI
