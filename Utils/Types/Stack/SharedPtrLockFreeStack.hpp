#pragma once
#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RenderAPI
{

template<typename T>
class OSharedPtrLockFreeStack
{
	struct SNode
	{
		OSharedPtr<T> Data;
		OAtomic<OSharedPtr<T>> Next;

		explicit SNode(const T& Other)
		    : Data(MakeShared(Other)) {}
	};

public:
	void Push(const T& Data);
	OSharedPtr<T> Pop();

	~OSharedPtrLockFreeStack()
	{
		while (Pop())
			;
	}

private:
	OAtomic<OSharedPtr<SNode>> Head;
};
template<typename T>
void OSharedPtrLockFreeStack<T>::Push(const T& Data)
{
	OSharedPtr<SNode> newNode = MakeShared(Data);
	newNode->Next = Head.load();
	while (!Head.compare_exchange_weak(newNode->Next, newNode))
		;
}

template<typename T>
OSharedPtr<T> OSharedPtrLockFreeStack<T>::Pop()
{
	OSharedPtr<SNode> oldHead = Head.load();
	while (oldHead && !Head.compare_exchange_weak(oldHead, oldHead->Next.load()))
		;

	if (oldHead)
	{
		oldHead->Next.store(OSharedPtr<T>());
		return oldHead->Data;
	}
}

} // namespace RenderAPI
