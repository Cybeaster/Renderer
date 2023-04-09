//
// Created by Cybea on 3/16/2023.
//

#ifndef RENDERAPI_THREADSAFELIST_HPP
#define RENDERAPI_THREADSAFELIST_HPP

#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RAPI
{

template<typename T>
class OThreadSafeList
{
	struct SNode
	{
		OMutex Mutex;
		OSharedPtr<T> Data;
		OUniquePtr<SNode> Next;

		SNode() = default;
		explicit SNode(const T& Other)
		    : Data(MakeShared<T>(Other)) {}
	};

public:
	OThreadSafeList(const OThreadSafeList& Other) = delete;
	OThreadSafeList& operator=(const OThreadSafeList& Other) = delete;

	OThreadSafeList() = default;

	~OThreadSafeList()
	{
		RemoveIf([](const SNode&)
		         { return true; });
	}

	void PushFront(const T& Value)
	{
		OUniquePtr<SNode> newNode(new SNode);
		OMutexGuard lock(Head.Mutex);
		newNode->Next = Move(Head.Next);
		Head.Next = Move(newNode);
	}

	template<typename FuncType>
	void ForEach(FuncType&& Function)
	{
		auto* current = &Head;
		OUniqueLock<OMutex> lock(Head.Mutex);
		while (const auto* nextNode = current->Next.get())
		{
			OUniqueLock<OMutex> nextLock(nextNode->Mutex);
			lock.unlock();
			Forward(Function)(*nextNode->Data);
			current = nextNode;
			lock = Move(nextLock);
		}
	}

	template<typename Pred>
	OSharedPtr<T> FindFirstIf(Pred&& Predicate)
	{
		auto* current = &Head;
		OUniqueLock<OMutex> lock(Head.Mutex);

		while (const auto* nextNode = current->Next.get())
		{
			OUniqueLock<OMutex> nextLock(nextNode->Mutex);
			lock.unlock();
			if (Forward(Predicate)(*nextNode->Data))
			{
				return nextNode->Data;
			}
			current = nextNode;
			lock = Move(nextLock);
		}
		return {};
	}

	template<typename Pred>
	void RemoveIf(Pred&& Predicate)
	{
		OUniqueLock<OMutex> lock(Head.Mutex);
		auto* current = &Head;
		while (const auto* nextNode = current->Next.get())
		{
			OUniqueLock<OMutex> nextLock(nextNode);
			if (Forward(Predicate)(*nextNode->Data))
			{
				auto tempNext = Move(current->Next);
				current->Next = Move(nextNode->Next);
				nextLock.unlock();
			}
			else
			{
				lock.unlock();
				lock = Move(nextLock);
				current = nextNode;
			}
		}
	}

private:
	SNode Head;
};

} // namespace RAPI

#endif // RENDERAPI_THREADSAFELIST_HPP
