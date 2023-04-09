#pragma once
#include "Functor/Functor.hpp"
#include "SmartPtr.hpp"
#include "Threads/HazardPointer.hpp"
#include "Utils/Types/Threads/Thread.hpp"
namespace RAPI
{

struct SDataToReclaim
{
	template<typename T>
	struct SRemover
	{
		static void Remove(void* Data)
		{
			delete static_cast<T>(Data);
		}
	};

	void* Data;
	SDataToReclaim* Next{ nullptr };
	OFunctor<void(void*)> Deleter;

	template<typename T>
	explicit SDataToReclaim(T* Pointer)
	    : Data(Pointer), Deleter(&SRemover<T>::Remove)
	{
	}

	~SDataToReclaim()
	{
		Deleter(Data);
	}
};

template<typename T>
class OLockFreeStackImpl
{
	struct SNode
	{
		OSharedPtr<T> Data;
		SNode* Next;
		explicit SNode(const T& NewData)
		    : Data(MakeShared(NewData)) {}
	};
};

template<typename T>
class OLockFreeStack : public OLockFreeStackImpl<T>
{
	// better to use hazard pointers

public:
	void Push(const T& Data);
	OSharedPtr<T> Pop();

private:
	//	void TryReclaim(SNode* OldHead);
	//	void DeleteNodes(SNode* Nodes);
	//	void ChainPendingNodes(SNode* First, SNode* Last);
	//	void ChainPendingNodes(SNode* Nodes);
	//	void ChainPendingNode(SNode* Node);

	void AddToReclaimList(SDataToReclaim* Node);

	template<typename DataType>
	void ReclaimLater(DataType* Data);

	void DeleteNonHazardNodes();

	// OAtomic<SNode*> GarbageNodes;

	OAtomic<SNode*> Head;
	OAtomic<uint32> NumThreadsInPop;

	OAtomic<SDataToReclaim*> NodesToReclaim;
};

template<typename T>
void OLockFreeStack<T>::DeleteNonHazardNodes()
{
	const SDataToReclaim* current = NodesToReclaim.exchange(nullptr); // claim all nodes for this current thread
	while (current)
	{
		auto* next = current->Next;

		if (!OHazardPointerManager::IsOutstandingHazardPointerFor(current->Data))
		{
			delete current;
		}
		else
		{
			AddToReclaimList(current);
		}

		current = next;
	}
}

template<typename T>
void OLockFreeStack<T>::AddToReclaimList(SDataToReclaim* Node)
{
	Node->Next = NodesToReclaim.load();
	while (!NodesToReclaim.compare_exchange_weak(Node->Next, Node))
		;
}
//
// template<typename T>
// void OLockFreeStack<T>::ChainPendingNode(OLockFreeStack::SNode* Node)
//{
//	ChainPendingNodes(Node, Node);
//}
//
// template<typename T>
// void OLockFreeStack<T>::ChainPendingNodes(OLockFreeStack::SNode* Nodes)
//{
//	auto* last = Nodes;
//	while (const auto* next = last->Next)
//	{
//		last = next;
//	}
//	ChainPendingNodes(Nodes, last);
//}
//
// template<typename T>
// void OLockFreeStack<T>::ChainPendingNodes(OLockFreeStack::SNode* First, OLockFreeStack::SNode* Last)
//{
//	Last->Next = GarbageNodes;
//	while (!GarbageNodes.compare_exchange_weak(Last->Next, First))
//		;
//}
//
// template<typename T>
// void OLockFreeStack<T>::DeleteNodes(OLockFreeStack::SNode* Nodes)
//{
//	while (Nodes)
//	{
//		auto* next = Nodes->Next;
//		delete Nodes;
//		Nodes = next;
//	}
//
//}
//
// template<typename T>
// void OLockFreeStack<T>::TryReclaim(SNode* OldHead)
//{
//	if (NumThreadsInPop == 1)
//	{
//		auto* nodesToDelete = GarbageNodes.exchange(nullptr);
//		if (!(--NumThreadsInPop))
//		{
//			DeleteNodes(nodesToDelete);
//		}
//		else if (nodesToDelete)
//		{
//			ChainPendingNodes(nodesToDelete);
//		}
//		delete OldHead;
//	}
//	else
//	{
//		ChainPendingNode(OldHead);
//		--NumThreadsInPop;
//	}
//}

/*
template<typename T>
OSharedPtr<T> OLockFreeStack<T>::Pop()
{
    ++NumThreadsInPop;
    auto* oldHead = Head.load();
    while (oldHead && !Head.compare_exchange_weak(oldHead, oldHead->Next))
        ;
    OSharedPtr<T> result;
    if (oldHead)
    {
        result.swap(oldHead->Data);
    }
    TryReclaim(oldHead);
    return result;
}
*/

template<typename T>
OSharedPtr<T> OLockFreeStack<T>::Pop()
{
	auto& hazardPointer = OHazardPointerManager::GetPointerForCurrentThread();
	SNode* oldHead = Head.load();

	do
	{
		SNode* tempNode;
		do // loop until you have set the hazard ptr to head
		{
			tempNode = oldHead;
			hazardPointer.store(oldHead); //  we have to ensure that the node hasn't been deleted
			oldHead = Head.load(); // keep looping to maintain the same value
		} while (oldHead != tempNode); // here we can say that hazard pointer references the correct oldHead
	} while (oldHead && !Head.compare_exchange_strong(oldHead, oldHead->Next)); // here we can say that oldHead and Head are the same
	// compare_exchange_strong is necessary because spurious failure on compare_exchange_weak will lead to setting hazardPointer pointlessly.

	hazardPointer.store(nullptr); // clear hp when we've finished
	OSharedPtr<T> result;

	if (oldHead)
	{
		result.swap(oldHead->Data);
		if (GetOutstandingHazardPointerFor(oldHead)) // check for hazard ptr ref a node before delete it
		{
			ReclaimLater(oldHead); // If you cannot delete it now, put into the list.
		}
		else
		{
			delete oldHead; // or delete it now
		}
		DeleteNonHazardNodes();
	}
	return result;
}

template<typename T>
void OLockFreeStack<T>::Push(const T& Data)
{
	auto newNode = new SNode(Data);
	newNode->Next = Head.load();
	while (!(Head.compare_exchange_weak(newNode->Next, newNode)))
		;
}

template<typename T>
template<typename DataType>
void OLockFreeStack<T>::ReclaimLater(DataType* Data)
{
	AddToReclaimList(new SDataToReclaim(Data));
}

} // namespace RAPI
