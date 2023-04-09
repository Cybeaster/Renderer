

#include "SmartPtr.hpp"
#include "TypeTraits.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RAPI
{

template<typename T>
class OReferenceBasedLockFreeStack
{
	struct SNode;
	struct SCountedNode
	{
		uint32 ExternalCount;
		SNode* Pointer;
	};

	struct SNode
	{
		OSharedPtr<T> Data;
		OAtomic<uint32> InternalCount{ 0 };
		SCountedNode Next;

		explicit SNode(const T& Other)
		    : Data(MakeShared(Other)) {}
	};

public:
	OReferenceBasedLockFreeStack()
	{
		static_assert(Head.is_always_lock_free, "This machine doesn't support DWCAS!");
	}
	~OReferenceBasedLockFreeStack()
	{
		while (Pop())
			;
	}

	OSharedPtr<T> Pop();

	void Push(const T& Data);
	void IncreaseHeadCount(SCountedNode& OldCounter);

private:
	OAtomic<SCountedNode> Head;
};
template<typename T>
OSharedPtr<T> OReferenceBasedLockFreeStack<T>::Pop()
{
	SCountedNode oldHead = Head.load(std::memory_order_relaxed);
	for (;;)
	{
		IncreaseHeadCount(oldHead); // started referencing head here
		const SNode* pointer = oldHead.Pointer; // once increased, start pointing to this node
		if (!pointer)
		{
			return {};
		}
		if (Head.compare_exchange_strong(oldHead, pointer->Next, std::memory_order_relaxed)) // try to remove oldHead
		{
			OSharedPtr<T> result;
			result.swap(pointer->Data);
			auto countIncrease = oldHead.ExternalCount - 2;

			if (pointer->InternalCount.fetch_add(countIncrease, std::memory_order_release) == -countIncrease)
			{
				delete pointer;
			}
			else if (pointer->InternalCount.fetch_add(-1, std::memory_order_relaxed) == 1)
			{
				pointer->InternalCount.load(std::memory_order_acquire);
				delete pointer;
			}
			return result;
		}

		if (pointer->InternalCount.fetch_sub(1) == 1)
		{
			delete pointer;
		}
	}
}

template<typename T>
void OReferenceBasedLockFreeStack<T>::IncreaseHeadCount(SCountedNode& OldCounter)
{
	SCountedNode newNode;
	do
	{
		newNode = OldCounter;
		++newNode.ExternalCount;
	} while (!Head.compare_exchange_strong(OldCounter, newNode, std::memory_order_acquire, std::memory_order_relaxed)); // try increase the counter, claiming this head for using
	OldCounter.ExternalCount = newNode.ExternalCount;
}

template<typename T>
void OReferenceBasedLockFreeStack<T>::Push(const T& Data)
{
	SCountedNode newNode;
	newNode.ptr = new SNode(Data);
	newNode.ExternalCount = 1;
	newNode.Pointer->Next = Head.load();
	// We may loop till load hasn't happened, so it's relaxed, but have to make store before popping.
	while (!Head.compare_exchange_weak(newNode.Pointer->Next, newNode, std::memory_order_release, std::memory_order_relaxed))
		;
}

} // namespace RAPI
