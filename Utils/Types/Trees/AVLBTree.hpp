#pragma once

#include "BSearchTree.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{

template<typename KeyType, typename ValueType>
class OAVLBTree : public OBSearchTree<KeyType, ValueType>
{
	struct SNode
	{
		KeyType Key;
		OSharedPtr<ValueType> Value;
		SNode* Right = nullptr;
		SNode* Left = nullptr;
		int8 Height = 0;
		SNode(KeyType NewKey, ValueType NewValue)
		    : Key(NewKey), Value(MakeShared(NewValue))
		{
		}
	};

public:
private:
	SNode* Remove(SNode* Node, KeyType Key);
	void Insert(SNode* Where, SNode* What) override;

	auto GetHeight(SNode* Node) const;
	void IncrementHeight(SNode* Node);
	int8 GetBalance(SNode* Node) const;
	void Swap(SNode* First, SNode* Second);
	void RotateRight(SNode* Node);
	void RotateLeft(SNode* Node);

	void Balance(SNode* Node);
};
template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::Insert(OAVLBTree::SNode* Where, OAVLBTree::SNode* What)
{
	// TODO Call super
	SNode* chosenNode = nullptr;
	if (What->Key < Where->Key)
	{
		chosenNode = &Where->Left;
	}
	else if (What->Key >= Where->Key)
	{
		chosenNode = &Where->Right;
	}

	if (chosenNode == nullptr)
	{
		chosenNode = What;
	}
	else
	{
		Insert(chosenNode, What);
	}

	IncrementHeight(Where);
	Balance(Where);
}

template<typename KeyType, typename ValueType>
OAVLBTree<KeyType, ValueType>::SNode* OAVLBTree<KeyType, ValueType>::Remove(OAVLBTree::SNode* Node, KeyType Key)
{
	if (Node == nullptr)
	{
		return nullptr;
	}

	if (Key < Node->Key)
	{
		Node->Left = Remove(Node->Left, Key);
	}
	else if (Key > Node->Key)
	{
		Node->Right = Remove(Node->Right, Key);
	}
	else
	{
		if (Node->Left == nullptr || Node->Right == nullptr)
		{
			Node = Node->Left == nullptr ? Node->Right : Node->Left;
		}
		else
		{
			SNode* maxNode = GetMax(Node->Left);
			Node->Key = Move(maxNode->Key);
			Node->Value = Move(maxNode->Value);
			Node->Left = Remove(Node->Left, maxNode->Key);
		}
	}

	if (Node != nullptr)
	{
		IncrementHeight(Node);
		Balance(Node);
	}

	return Node;
}

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::Balance(OAVLBTree::SNode* Node)
{
	auto balance = GetBalance(Node);

	if (balance == -2)
	{
		if (GetBalance(Node->Left) == 1)
		{
			RotateLeft(Node->Left);
		}
		RotateRight(Node);
	}
	else if (balance == 2)
	{
		if (GetBalance(Node->Right == -1))
		{
			RotateRight(Node->Right);
		}
		RotateLeft(Node);
	}
}

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::RotateLeft(OAVLBTree::SNode* Node)
{
	Swap(Node, Node->Right);

	auto buffer = Node->Left;

	Node->Left = Node->Right;
	Node->Right = Node->Left->Right;

	Node->Right->Left = Node->Right->Right;
	Node->Left->Right = Node->Left->Left;

	Node->Left->Left = buffer;

	IncrementHeight(Node->Left);
	IncrementHeight(Node->Right);
}

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::RotateRight(OAVLBTree::SNode* Node)
{
	Swap(Node, Node->Left);
	auto buffer = Node->Right;

	Node->Right = Node->Left;
	Node->Left = Node->Right->Left;

	Node->Right->Left = Node->Right->Right;
	Node->Right->Right = buffer;

	IncrementHeight(Node->Right);
	IncrementHeight(Node);
}

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::Swap(OAVLBTree::SNode* First, OAVLBTree::SNode* Second)
{
	const auto firstKey = First->Key;
	const auto firstValue = First->Value;

	First->Value = Move(Second->Value);
	First->Key = Move(Second->Key);

	Second->Key = Move(firstKey);
	Second->Value = Move(firstValue);
}

template<typename KeyType, typename ValueType>
int8 OAVLBTree<KeyType, ValueType>::GetBalance(SNode* Node) const
{
	return Node == nullptr ? 0 : GetHeight(Node->Right) - GetHeight(Node->Left);
}
template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::IncrementHeight(OAVLBTree::SNode* Node)
{
	Node->Height = SMath::Max(GetHeight(Node->Left), GetHeight(Node->Right)) + 1;
}

template<typename KeyType, typename ValueType>
auto OAVLBTree<KeyType, ValueType>::GetHeight(OAVLBTree::SNode* Node) const
{
	return Node == nullptr ? -1 : Node->Height;
}
} // namespace RAPI