#pragma once

#include "BSearchTreeBase.hpp"
namespace RAPI
{

template<typename KeyType, typename ValueType>
class ORedBlackTree
{
	enum class EColor
	{
		Black,
		Red
	};

	struct SNode
	{
		KeyType Key;
		ValueType Value;
		SNode* Left;
		SNode* Right;
		SNode* Parent;
		EColor Color;

		SNode(KeyType KeyParam, ValueType ValueParam, SNode* NullNode = nullptr)
		{
			Key = KeyParam;
			Value = ValueParam;
			Color = EColor::Red;

			Left = NullNode;
			Right = NullNode;
			Parent = NullNode;
		}
	};

	void Insert(KeyType Key, ValueType Value);

private:
	SNode* NullNode = new SNode();
	SNode* Head = nullptr;

	void Balance(SNode* Node);
	void Remove(KeyType Key);
	bool DoesNodeExist(SNode* Node);
};

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Remove(KeyType Key)
{

}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Balance(SNode* Node)
{
	SNode* uncle = nullptr;

	while (Node->Parent->Color == EColor::Red)
	{
		if (Node->Parent == Node->Parent->Parent->Left) // Check which side we are working with
		{
			uncle = Node->Parent->Parent->Right;
			if (uncle->Color == EColor::Red)
			{
				Node->Parent->Color = EColor::Black;
				uncle->Color = EColor::Black;

				Node->Parent->Parent->Color = EColor::Red;
				Node = Node->Parent->Parent;
			}
			else
			{
				if (Node == Node->Parent->Right)
				{
					Node = Node->Parent;
					OBTreeUtils::RotateLeft(Node);
				}
				Node->Parent->Color = EColor::Black;
				Node->Parent->Parent->Color = EColor::Red;


				OBTreeUtils::RotateRight(Node->Parent->Parent);
			}
		}
		else
		{
			uncle = Node->Parent->Parent->Left;

			if (uncle->Color == EColor::Red)
			{
				Node->Parent->Color = EColor::Black;
				uncle->Color = EColor::Black;

				Node->Parent->Parent->Color = EColor::Red;
				Node = Node->Parent->Parent;
			}
			else
			{
				if (Node == Node->Parent->Left)
				{
					Node = Node->Parent;
					OBTreeUtils::RotateRight(Node);
				}
				Node->Parent->Color = EColor::Black;
				Node->Parent->Parent->Color = EColor::Red;
				OBTreeUtils::RotateLeft(Node->Parent->Parent);
			}
		}
	}
	Head->Color = EColor::Black;
}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Insert(KeyType Key, ValueType Value)
{
	auto currentNode = Head;
	auto parent = NullNode;

	while (DoesNodeExist(currentNode))
	{
		parent = currentNode;
		if (Value < currentNode->Value)
		{
			currentNode = currentNode->Left;
		}
		else
		{
			currentNode = currentNode->Right;
		}
	}
	auto newNode = SNode(Key, Value, NullNode);
	newNode.Parent = parent;

	if (parent == NullNode)
	{
		Head = newNode;
	}
	else if (Value < parent->Value)
	{
		parent->Left = newNode;
	}
	else
	{
		parent->Right = newNode;
	}

	Balance(newNode);
}

template<typename KeyType, typename ValueType>
bool ORedBlackTree<KeyType, ValueType>::DoesNodeExist(ORedBlackTree::SNode* Node)
{
	Node != NullNode;
}

} // namespace RAPI