#pragma once

#include "Assert.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{

template<typename TKeyType, typename TValueType>
struct SBSearchTreeNode
{
	TKeyType Key;
	OSharedPtr<TValueType> Value;
	SBSearchTreeNode* Right = nullptr;
	SBSearchTreeNode* Left = nullptr;
	SBSearchTreeNode(TKeyType NewKey, TValueType NewValue)
	    : Key(NewKey), Value(MakeShared(NewValue))
	{
	}
};

template<typename TKeyType, typename TValueType, typename TNode = SBSearchTreeNode<TKeyType, TValueType>>
class OBSearchTree
{
public:
	enum ETraverseType
	{
		Symmetric, // left child -> parent -> right child
		Reverse,
		Direct
	};

	virtual void Insert(TKeyType Key, TValueType Value);
	TValueType Find(TKeyType Key);
	TValueType GetMax();
	TValueType GetMin();
	bool Remove(TKeyType Key);
	void Print(OPrinter* Printer);

private:
	void Print(TNode* Where, OBSearchTree::ETraverseType Type, OPrinter* Printer);

	template<typename Operation>
	void TraverseTreeWith(TNode* Where, ETraverseType Type, Operation Op);

	TNode* GetMax(TNode* Where);
	TNode* Remove(TNode* Where, TKeyType What);
	TNode* GetMin(TNode* Where);
	TValueType Find(TNode* Where, TKeyType What);
	void Insert(TNode* Where, TNode* What);

	OSharedPtr<TNode> Head = nullptr;
};

template<typename TKeyType, typename TValueType, typename TNode>
template<typename Operation>
void OBSearchTree<TKeyType, TValueType, TNode>::TraverseTreeWith(TNode* Where, OBSearchTree::ETraverseType Type, Operation Op)
{
	if (Where == nullptr)
	{
		return;
	}

	switch (Type)
	{
	case ETraverseType::Symmetric:
	{
		TraverseTreeWith(Where->Left, Type, Op);
		Op(Where);
		TraverseTreeWith(Where->Right, Type, Op);
		break;
	}
	case ETraverseType::Reverse:
	{
		TraverseTreeWith(Where->Left, Type, Op);
		TraverseTreeWith(Where->Right, Type, Op);
		Op(Where);
	}
	case ETraverseType::Direct:
	{
		Op(Where);
		TraverseTreeWith(Where->Left, Type, Op);
		TraverseTreeWith(Where->Right, Type, Op);
	}
	}
}

template<typename TKeyType, typename TValueType, typename TNode>
void OBSearchTree<TKeyType, TValueType, TNode>::Print(TNode* Where, OBSearchTree::ETraverseType Type, OPrinter* Printer)
{
	TraverseTreeWith(Where, Type, [Printer](TNode* Node)
	                 { *Printer << "Key: " << Node->Key << "Value: " << Node->Value << "\n"; });
}

template<typename TKeyType, typename TValueType, typename TNode>
void OBSearchTree<TKeyType, TValueType, TNode>::Print(OBSearchTree::ETraverseType Type, OPrinter* Printer)
{
	Print(Head, Type, Printer);
}

template<typename TKeyType, typename TValueType, typename TNode>
TNode* OBSearchTree<TKeyType, TValueType, TNode>::GetMax(TNode* Where)
{
	if (Where->Right != nullptr)
	{
		return Where->Right;
	}
	else
	{
		return Where;
	}
}

template<typename TKeyType, typename TValueType, typename TNode>
TNode* OBSearchTree<TKeyType, TValueType, TNode>::Remove(TNode* Where, TKeyType What)
{
	if (Where == nullptr)
	{
		return nullptr;
	}

	if (Where->Key > What)
	{
		return Where->Right = Remove(Where->Right, What);
	}
	else if (Where->Key < What)
	{
		return Where->Left = Remove(Where->Left, What);
	}
	else
	{
		if (Where->Right == nullptr || Where->Left == nullptr)
		{
			Where = Where->Right == nullptr ? Where->Left :
			                                  Where->Right;
		}
		else
		{
			auto maxNode = GetMax(Where->Left);
			Where->Key = Move(maxNode->Key);
			Where->Value = Move(maxNode->Value);
			Where->Left = Remove(Where->Left, maxNode.Key);
		}
	}

	return Remove(Where, What);
} // namespace RAPI

template<typename TKeyType, typename TValueType, typename TNode>
bool OBSearchTree<TKeyType, TValueType, TNode>::Remove(TKeyType Key)
{
	auto node = Remove(Head, Key);
	if (node)
	{
		delete node;
		return true;
	}
	return false;
}

template<typename TKeyType, typename TValueType, typename TNode>
TNode* OBSearchTree<TKeyType, TValueType, TNode>::GetMin(TNode* Where)
{
	if (Where->Left == nullptr)
	{
		return Where->Left;
	}

	return GetMin(Where->Left);
}

template<typename TKeyType, typename TValueType, typename TNode>
TValueType OBSearchTree<TKeyType, TValueType, TNode>::GetMin()
{
	if (ENSURE(Head != nullptr))
	{
		return GetMin(Head)->Value;
	}
	return TValueType();
}

template<typename TKeyType, typename TValueType, typename TNode>
TValueType OBSearchTree<TKeyType, TValueType, TNode>::GetMax()
{
	if (ENSURE(Head != nullptr))
	{
		return GetMax(Head).Value;
	}
	return TValueType();
}

template<typename TKeyType, typename TValueType, typename TNode>
TValueType OBSearchTree<TKeyType, TValueType, TNode>::Find(TNode* Where, TKeyType What)
{
	if (ENSURE(Where != nullptr))
	{
		if (Where->Key == What)
		{
			return Where->Value;
		}

		return Where->Key > What ? Find(Where->Left, What) : Find(Where->Right, What);
	}
	return TValueType();
}

template<typename TKeyType, typename TValueType, typename TNode>
TValueType OBSearchTree<TKeyType, TValueType, TNode>::Find(TKeyType Key)
{
	if (ENSURE(Head != nullptr))
	{
		if (Head->Key == Key)
		{
			return Head->Value;
		}

		return Find(Head, Key);
	}
}

template<typename TKeyType, typename TValueType, typename TNode>
void OBSearchTree<TKeyType, TValueType, TNode>::Insert(TNode* Where, TNode* What)
{
	if (What->Key > Where->Key)
	{
		if (Where->Right == nullptr)
		{
			Where->Right = Where;
			return;
		}
	}
	else
	{
		if (Where->Left == nullptr)
		{
			Where->Left = Where;
			return;
		}
	}
	Insert(Where, What);
}

template<typename TKeyType, typename TValueType, typename TNode>
void OBSearchTree<TKeyType, TValueType, TNode>::Insert(TKeyType Key, TValueType Value)
{
	auto newNode = new TNode(Key, Value);
	if (Head == nullptr)
	{
		Head = new TNode(Key, Value);
	}
	else
	{
		Insert(Head, newNode);
	}
}

} // namespace RAPI