#pragma once
#include "BSearchTreeBase.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{

template<typename KeyType, typename ValueType>
struct SAVLTreeNode
{
	KeyType Key;
	OSharedPtr<ValueType> Value;
	SAVLTreeNode* Right = nullptr;
	SAVLTreeNode* Left = nullptr;
	int8 Height = 0;
	SAVLTreeNode(KeyType NewKey, ValueType NewValue)
	    : Key(NewKey), Value(MakeShared(NewValue))
	{
	}
};

template<typename KeyType, typename ValueType>
class OAVLBTree : public OBSearchTreeBase<KeyType, ValueType, SAVLTreeNode<KeyType, ValueType>>
{
	using NodeType = SAVLTreeNode<KeyType, ValueType>;

public:
	bool Remove(KeyType Key) override;
	void Insert(KeyType Key, ValueType Value) override;

private:
	NodeType* Remove(NodeType* Node, KeyType Key);
	void Insert(NodeType* Where, NodeType* What);
	void Balance(NodeType* Node);
};

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::Insert(KeyType Key, ValueType Value)
{
	Insert(Root, Key, Value);
}

template<typename KeyType, typename ValueType>
void OAVLBTree<KeyType, ValueType>::Insert(OAVLBTree::NodeType* Where, OAVLBTree::NodeType* What)
{
	OBTreeUtils::Insert(Where, What);

	IncrementHeight(Where);
	Balance(Where);
}

template<typename KeyType, typename ValueType>
OAVLBTree<KeyType, ValueType>::NodeType* OAVLBTree<KeyType, ValueType>::Remove(OAVLBTree::NodeType* Node, KeyType Key)
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
			NodeType* maxNode = GetMax(Node->Left);
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
void OAVLBTree<KeyType, ValueType>::Balance(OAVLBTree::NodeType* Node)
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

} // namespace RAPI