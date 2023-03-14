#pragma once
#include "SmartPtr.hpp"
#include "Thread.hpp"

#include <queue>

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

	OSharedPtr<T> TryPop()
	{
		OUniquePtr<SNode> oldHead = PopHead();
		return oldHead ? oldHead->Data : nullptr;
	}

	template<typename Arg>
	void Push(Arg&& Value)
	{
		OSharedPtr<Arg> value = MakeShared(Move(Value));
		OUniquePtr<SNode> newValue(new SNode);
		SNode* const newNode = newValue.get();

		OMutexGuard lock(TailMutex);
		Tail->Data = value;
		Tail->Next = Move(newValue);
		Tail = newNode;
	}

private:
	SNode* GetTail()
	{
		OMutexGuard lock(TailMutex);
		return Tail;
	}

	OUniquePtr<SNode> PopHead()
	{
		OMutexGuard lock(HeadMutex);
		if (Head.get() == GetTail())
		{
			return nullptr;
		}

		OUniquePtr<SNode> oldHead = Move(Head);
		Head = Move(oldHead->Next);
		return oldHead;
	}

	OUniquePtr<SNode> Head;
	SNode* Tail;
	OMutex HeadMutex;
	OMutex TailMutex;
};

template<typename T>
class OThreadSafeQueueSlow
{
public:
	OThreadSafeQueueSlow() = default;

	template<class Type>
	void Push(Type&& Argument)
	{
		auto value = MakeShared(Move(Argument));

		OMutexGuard lock(Mutex);
		Data.push(Move(value));
		DataCondition.notify_one();
	}

	void WaitAndPop(T& Argument)
	{
		OUniqueLock lock(Mutex);
		DataCondition.wait(lock, [this]
		                   { return !Data.empty(); });

		Argument = Move(*Data.front());
		Data.pop();
	}

	OSharedPtr<T> WaitAndPop()
	{
		OUniqueLock lock(Mutex);
		DataCondition.wait(lock, [this]
		                   { return !Data.empty(); });

		auto result = Move(Data.front());
		Data.pop();
		return result;
	}

	bool TryPop(T& Argument)
	{
		OMutexGuard lock(Mutex);
		if (Data.empty())
		{
			return false;
		}
		Argument = Move(*Data.front());
		Data.pop();
		return true;
	}

	OSharedPtr<T> TryPop()
	{
		OMutexGuard lock(Mutex);
		if (Data.empty())
		{
			return {};
		}
		auto result = Move(Data.front());
		Data.pop();
		return result;
	}

	bool Empty()
	{
		OMutexGuard lock(Mutex);
		return Data.empty();
	}

private:
	mutable OMutex Mutex;
	OQueue<OSharedPtr<T>> Data;
	OConditionVariable DataCondition;
};
} // namespace RenderAPI