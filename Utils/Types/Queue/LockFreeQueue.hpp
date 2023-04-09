//
// Created by Cybea on 3/20/2023.
//

#ifndef RENDERAPI_LOCKFREEQUEUE_HPP
#define RENDERAPI_LOCKFREEQUEUE_HPP

#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RAPI
{

template<typename T>
class OLockFreeQueue
{
	struct SNode;

	struct SCountedNode
	{
		int32 ExternalCount;
		SNode* Pointer;
	};

	struct SNodeCounter
	{
		uint32 InternalCount : 30;
		uint32 ExternalCount : 2;
	};

	struct SNode
	{
		OAtomic<T*> Data;
		OAtomic<SNodeCounter> Count;
		OAtomic<SCountedNode> Next;
		SNode()
		{
			SNodeCounter newCount;
			newCount.InternalCount = 0;
			newCount.ExternalCount = 2; // cuz its referenced from tail and previous node
			Count.store(newCount);
			Next.Pointer = nullptr;
			Next.ExternalCount = 0;
		}

		void ReleaseReference();
	};

public:
	OLockFreeQueue()
	    : Head(new SNode), Tail(Head.load()) {}
	REMOVE_COPY_FOR(OLockFreeQueue)

	~OLockFreeQueue()
	{
		while (const SNode* oldHead = Head.load())
		{
			Head.store(oldHead->Next);
			delete oldHead;
		}
	}

	OSharedPtr<T> Pop();
	void Push(const T& Value);

private:
	void IncreaseExternalCount(OAtomic<SCountedNode>& Node, SCountedNode& OldNode);
	void FreeExternalCounter(SCountedNode*);
	void SetNewTail(SCountedNode& OldTail, const SCountedNode& NewTail);

	OAtomic<SCountedNode> Head;
	OAtomic<SCountedNode> Tail;
};

template<typename T>
void OLockFreeQueue<T>::SetNewTail(OLockFreeQueue::SCountedNode& OldTail, const OLockFreeQueue::SCountedNode& NewTail)
{
	const SNode* currentTailPtr = OldTail.Pointer;
	while (!Tail.compare_exchange_weak(OldTail, NewTail) && OldTail.Pointer == currentTailPtr)
		;
	if (OldTail.Pointer == currentTailPtr)
	{
		FreeExternalCounter(OldTail);
	}
	else
	{
		currentTailPtr->ReleaseReference();
	}
}

template<typename T>
void OLockFreeQueue<T>::FreeExternalCounter(SCountedNode* OldNode)
{
	auto pointer = OldNode->Pointer;
	int32 countIncrease = OldNode->ExternalCount - 2;
	SNodeCounter oldCounter = pointer->Count.load(std::memory_order_relaxed);
	SNodeCounter newCounter;
	do
	{
		newCounter = oldCounter;
		--newCounter.ExternalCount;
		newCounter.InternalCount += countIncrease;
	} while (!pointer->Count.compare_exchange_strong(oldCounter, newCounter, std::memory_order_acquire, std::memory_order_relaxed));

	if (!newCounter.InternalCount && !newCounter.ExternalCount)
	{
		delete pointer;
	}
}

template<typename T>
void OLockFreeQueue<T>::IncreaseExternalCount(OAtomic<SCountedNode>& Node, SCountedNode& OldNode)
{
	SCountedNode newCounter;
	do
	{
		newCounter = OldNode;
		++newCounter.ExternalCount;
	} while (Node.compare_exchange_strong(OldNode, newCounter, std::memory_order_acquire, std::memory_order_relaxed));
	OldNode.ExternalCount = newCounter.ExternalCount;
}

template<typename T>
void OLockFreeQueue<T>::SNode::ReleaseReference()
{
	SNodeCounter oldCounter = Count.load(std::memory_order_relaxed);
	SNodeCounter newCounter;
	do
	{
		newCounter = oldCounter;
		--newCounter.InternalCount;
	} while (!Count.compare_exchange_strong(oldCounter, newCounter, std::memory_order_acquire, std::memory_order_relaxed));

	if (!newCounter.InternalCount && !newCounter.ExternalCount)
	{
		delete this;
	}
}

template<typename T>
void OLockFreeQueue<T>::Push(const T& Value)
{
	OUniquePtr<T> newData(new T(Value));
	SCountedNode newNext;
	newNext.Pointer = new SNode;
	newNext.ExternalCount = 1;

	SCountedNode oldTail = Tail.load();

	while (true)
	{
		IncreaseExternalCount(Tail, oldTail);
		T* oldData = nullptr;

		if (oldTail.Pointer->Data.compare_exchange_strong(oldData, newData.get())) // If set, need to handle where other thread has helped this one
		{
			SCountedNode oldNext = { 0 };
			if (!oldTail.Pointer->Next.compare_exchange_strong(oldNext, newNext)) // if fails, then another thread has already set the next pointer.
			{
				delete newNext.Pointer;
				newNext = oldNext; // Use the value that is set by other thread.
			}
			SetNewTail(oldTail, newNext);
			newData.release();
			break;
		}
		else // If couldn't, we need help //NOLINT
		{
			SCountedNode oldNext = { 0 };
			if (oldTail.Pointer->Next.compare_exchange_strong(oldNext, newNext))
			{
				oldNext = newNext;
				newNext.Pointer = new SNode();
			}
			SetNewTail(oldTail, oldNext);
		}
	}
}

template<typename T>
OSharedPtr<T> OLockFreeQueue<T>::Pop()
{
	SCountedNode* oldHead = Head.load(std::memory_order_relaxed);
	while (true)
	{
		IncreaseExternalCount(Head, oldHead);
		SNode* oldData = oldHead->Pointer;
		if (oldData == Tail.load().Pointer)
		{
			oldData->ReleaseReference();
			return {};
		}
		SCountedNode next = oldData->Next.load();
		if (Head.compare_exchange_strong(oldHead, oldData->Next))
		{
			const T* result = oldData->Data.exchange(nullptr);
			FreeExternalCounter(oldHead);
			return std::make_unique(result);
		}
		oldData->ReleaseReference();
	}
}

} // namespace RAPI

#endif // RENDERAPI_LOCKFREEQUEUE_HPP
