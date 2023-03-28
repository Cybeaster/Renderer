#pragma once
#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"

namespace RenderAPI
{

/**
 * @brief Well grained queue allowing thread-safe operations*/
template<typename T>
class OThreadSafeQueue
{
	struct SNode
	{
		OSharedPtr<T> Data;
		OUniquePtr<T> Next;
	};

public:
	OThreadSafeQueue()
	    : Head(new SNode), Tail(Head.get()) {}

	OThreadSafeQueue(const OThreadSafeQueue& Other) = delete;
	OThreadSafeQueue& operator=(const OThreadSafeQueue& Other) = delete;

	OSharedPtr<T> TryPop();

	bool TryPop(T& Value);

	template<typename Arg>
	void Push(Arg&& Value);

	void WaitAndPop(T& Argument);
	OSharedPtr<T> WaitAndPop();

	bool IsEmpty();

private:
	SNode* GetTail();
	OUniquePtr<SNode> PopHead();

	/**@note Returns the same lock to ensure that the same one is held. */
	OUniqueMutexLock WaitForData();

	OUniquePtr<SNode> WaitPopHead();
	OUniquePtr<SNode> WaitPopHead(T& Value);

	OUniquePtr<SNode> TryPopHead();
	OUniquePtr<SNode> TryPopHead(T& Value);

	OUniquePtr<SNode> Head;
	SNode* Tail;
	OMutex HeadMutex;
	OMutex TailMutex;
	OConditionVariable DataCondition;
};

template<typename T>
bool OThreadSafeQueue<T>::IsEmpty()
{
	OMutexGuard lock(HeadMutex);
	return Head.get() == Tail;
}

template<typename T>
OUniquePtr<typename OThreadSafeQueue<T>::SNode> OThreadSafeQueue<T>::WaitPopHead()
{
	OUniqueMutexLock lock(WaitForData());
	return PopHead();
}

template<typename T>
OUniquePtr<typename OThreadSafeQueue<T>::SNode> OThreadSafeQueue<T>::WaitPopHead(T& Value)
{
	OUniqueMutexLock(WaitForData());
	Value = Move(*Head->Data);
	return PopHead();
}

template<typename T>
OUniqueMutexLock OThreadSafeQueue<T>::WaitForData()
{
	OUniqueMutexLock headLock(HeadMutex);
	DataCondition.wait(headLock, [&]
	                   { Head.get() != GetTail(); });
	return Move(headLock);
}

template<typename T>
OSharedPtr<T> OThreadSafeQueue<T>::TryPop()
{
	OUniquePtr<SNode> oldHead = PopHead();
	return oldHead ? oldHead->Data : nullptr;
}

template<typename T>
typename OThreadSafeQueue<T>::SNode* OThreadSafeQueue<T>::GetTail()
{
	OMutexGuard lock(TailMutex);
	return Tail;
}

template<typename T>
OUniquePtr<typename OThreadSafeQueue<T>::SNode> OThreadSafeQueue<T>::PopHead()
{
	OUniquePtr<SNode> oldHead = Move(Head);
	Head = Move(oldHead->Next);
	return oldHead;
}

template<typename T>
template<typename Arg>
void OThreadSafeQueue<T>::Push(Arg&& Value)
{
	OSharedPtr<Arg> value = MakeShared(Move(Value));

	OUniquePtr<SNode> newValue(new SNode);
	SNode* const newNode = newValue.get();
	{
		OMutexGuard lock(TailMutex);
		Tail->Data = value;
		Tail->Next = Move(newValue);
		Tail = newNode;
	}
	DataCondition.notify_one();
}

template<typename T>
OSharedPtr<T> OThreadSafeQueue<T>::WaitAndPop()
{
	OUniquePtr<SNode> oldHead = WaitPopHead();
	return oldHead->Data;
}

template<typename T>
void OThreadSafeQueue<T>::WaitAndPop(T& OutValue)
{
	OUniquePtr<SNode> const oldHead = WaitPopHead(OutValue);
}

template<typename T>
OUniquePtr<typename OThreadSafeQueue<T>::SNode> OThreadSafeQueue<T>::TryPopHead()
{
	OMutexGuard lock(HeadMutex);
	if (Head.get() == GetTail())
	{
		return {};
	}
	return PopHead();
}

template<typename T>
OUniquePtr<typename OThreadSafeQueue<T>::SNode> OThreadSafeQueue<T>::TryPopHead(T& Value)
{
	OMutexGuard lock(HeadMutex);
	if (Head.get() == GetTail())
	{
		return {};
	}
	Value = Move(*Head->Data);
	return PopHead();
}

template<typename T>
bool OThreadSafeQueue<T>::TryPop(T& Value)
{
	OUniquePtr<SNode> oldHead = TryPopHead(Value);
	return oldHead;
}

} // namespace RenderAPI