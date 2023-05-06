#pragma once

#include "Assert.hpp"
#include "BSearchTreeBase.hpp"
#include "SmartPtr.hpp"

namespace RAPI
{

template<typename KeyType, typename ValueType>
struct SBSearchTreeNode
{
	KeyType Key;
	OSharedPtr<ValueType> Value;
	SBSearchTreeNode* Right = nullptr;
	SBSearchTreeNode* Left = nullptr;
	SBSearchTreeNode(KeyType NewKey, ValueType NewValue)
	    : Key(NewKey), Value(MakeShared(NewValue))
	{
	}
};

template<typename KeyType, typename ValueType>
class OBSearchTree : OBSearchTreeBase<KeyType, ValueType, SBSearchTreeNode<KeyType, ValueType>>
{
	using TNode = SBSearchTreeNode<KeyType, ValueType>;

public:

	void Insert(KeyType Key, ValueType Value) override;
	bool Remove(KeyType Key) override;

private:
	TNode* Remove(TNode* Where, KeyType Key);
};

template<typename KeyType, typename ValueType>
void OBSearchTree<KeyType, ValueType>::Insert(KeyType Key, ValueType Value)
{
	OBTreeUtils::Insert(Root, Key, Value);
}

template<typename KeyType, typename ValueType>
OBSearchTree<KeyType, ValueType>::TNode* OBSearchTree<KeyType, ValueType>::Remove(OBSearchTree::TNode* Where, KeyType Key)
{
	if (Where == nullptr)
	{
		return nullptr;
	}

	if (Where->Key > Key)
	{
		return Where->Right = Remove(Where->Right, Key);
	}
	else if (Where->Key < Key)
	{
		return Where->Left = Remove(Where->Left, Key);
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
}

template<typename KeyType, typename ValueType>
bool OBSearchTree<KeyType, ValueType>::Remove(KeyType Key)
{
	auto node = Remove(Root, Key);
	if (node)
	{
		delete node;
		return true;
	}
	return false;
}

} // namespace RAPI