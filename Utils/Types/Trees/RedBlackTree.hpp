#pragma once

#include "BSearchTreeBase.hpp"
namespace RAPI
{

enum class ERedBlackTreeColor
{
	Black,
	Red
};

template<typename KeyType, typename ValueType>
struct SRedBlackTreeNode
{
	using TColor = ERedBlackTreeColor;

	KeyType Key;
	ValueType Value;
	SRedBlackTreeNode* Left;
	SRedBlackTreeNode* Right;
	SRedBlackTreeNode* Parent;
	TColor Color : 2;

	SRedBlackTreeNode(KeyType KeyParam, ValueType ValueParam, SRedBlackTreeNode* NullNode = nullptr)
	{
		Key = KeyParam;
		Value = ValueParam;
		Color = TColor::Red;

		Left = NullNode;
		Right = NullNode;
		Parent = NullNode;
	}
};

template<typename KeyType, typename ValueType>
class ORedBlackTree : OBSearchTreeBase<KeyType, ValueType, SRedBlackTreeNode<KeyType, ValueType>>
{
	using TColor = ERedBlackTreeColor;
	using NodeType = SRedBlackTreeNode<KeyType, ValueType>;

	void Insert(KeyType Key, ValueType Value) override;
	void Remove(KeyType Key) override;

private:
	NodeType* NullNode = new NodeType();

	void Balance(NodeType* Node);
	bool DoesNodeExist(NodeType* Node);
	NodeType* GetChildOrMock(NodeType* Node);
	uint8 GetChildrenCount(NodeType* Node);
	void TransplantNode(NodeType* FromNode, NodeType* ToNode);
	void FixRulesAfterRemoval(NodeType* Node);
};

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::FixRulesAfterRemoval(ORedBlackTree::NodeType* Node)
{
	using TColor::Black;
	using TColor::Red;

	while (Node != Root && Node->Color == Black)
	{
		NodeType* brother = nullptr;

		if (Node == Node->Parent->Left)
		{
			brother = Node->Parent->Right;
			if (brother->Color == Red)
			{
				brother->Color = Black;
				Node->Parent->Color = Black;

				Node->Parent->Color = Red;
				OBTreeUtils::RotateLeft(Node->Parent);
			}

			if (brother->Left->Color == TColor::Black && brother->Right->Color == Black)
			{
				brother->Color = Red;
				Node = Node->Parent;
			}
			else
			{
				if (brother->Right->Color == Black)
				{
					brother->Left->Color = TColor::Black;
					brother->Color = Red;
					OBTreeUtils::RotateRight(brother);
					brother = Node->Parent->Right;
				}

				brother->Color = Node->Parent->Color;
				Node->Parent->Color = Black;
				brother->Right->Color = Black;

				OBTreeUtils::RotateLeft(Node->Parent);

				Node = Root;
			}
		}
		else
		{
			brother = Node->Parent->Left;

			if (brother->Color == Red)
			{
				brother->Color = Black;
				Node->Parent->Color = Red;
				OBTreeUtils::RotateRight(Node);
				brother = Node->Parent->Left;
			}
			if (brother->Left->Color == Black && brother->Right->Color == Black)
			{
				OBTreeUtils::RotateRight(Node->Parent);
				brother = Node->Parent->Left;
			}
			else
			{
				if (brother->Left->Color == Black)
				{
					brother->Right->Color == Black;
					brother->Color = Red;

					OBTreeUtils::RotateLeft(brother);
					brother = Node->Parent->Left;
				}

				brother->Color = Node->Parent->Color;
				brother->Parent->Color = Black;
				brother->Left->Color = Black;
				OBTreeUtils::RotateRight(Node->Parent);

				Node = Root;
			}
		}
	}
	Node->Color = Black;
}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::TransplantNode(ORedBlackTree::NodeType* FromNode, ORedBlackTree::NodeType* ToNode)
{
	if (ToNode == Root)
	{
		Root = FromNode;
	}
	else if (ToNode == ToNode->Parent->Left)
	{
		ToNode->Parent->Left = FromNode;
	}
	else
	{
		ToNode->Parent->Right = FromNode;
	}
	FromNode->Parent = ToNode->Parent;
}

template<typename KeyType, typename ValueType>
uint8 ORedBlackTree<KeyType, ValueType>::GetChildrenCount(ORedBlackTree::NodeType* Node)
{
	uint8 count = 0;
	if (DoesNodeExist(Node->Left))
	{
		count++;
	}
	if (DoesNodeExist(Node->Right))
	{
		count++;
	}
	return count;
}

template<typename KeyType, typename ValueType>
ORedBlackTree<KeyType, ValueType>::NodeType* ORedBlackTree<KeyType, ValueType>::GetChildOrMock(ORedBlackTree::NodeType* Node)
{
	return DoesNodeExist(Node->Left) ? Node->Left : Node->Right;
}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Remove(KeyType Key)
{
	auto nodeToDelete = OBTreeUtils::FindNode(Root, Key);
	auto removedNodeColor = nodeToDelete->Color;

	NodeType* child = nullptr;

	if (GetChildrenCount(nodeToDelete) < 2)
	{
		child = GetChildOrMock(nodeToDelete);
		TransplantNode(nodeToDelete, child);
	}
	else
	{
		NodeType* minNode = OBTreeUtils::GetMin(nodeToDelete);
		nodeToDelete->Key = Move(minNode->Key);
		nodeToDelete->Value = Move(minNode->Value);
		removedNodeColor = minNode->Color;
		child = GetChildOrMock(minNode);
		TransplantNode(minNode, child);
	}
	if (removedNodeColor == TColor::Black)
	{
		FixRulesAfterRemoval(child);
	}
}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Balance(NodeType* Node)
{
	NodeType* uncle = nullptr;

	while (Node->Parent->Color == TColor::Red)
	{
		if (Node->Parent == Node->Parent->Parent->Left) // Check which side we are working with
		{
			uncle = Node->Parent->Parent->Right;
			if (uncle->Color == TColor::Red)
			{
				Node->Parent->Color = TColor::Black;
				uncle->Color = TColor::Black;

				Node->Parent->Parent->Color = TColor::Red;
				Node = Node->Parent->Parent;
			}
			else
			{
				if (Node == Node->Parent->Right)
				{
					Node = Node->Parent;
					OBTreeUtils::RotateLeft(Node);
				}
				Node->Parent->Color = TColor::Black;
				Node->Parent->Parent->Color = TColor::Red;

				OBTreeUtils::RotateRight(Node->Parent->Parent);
			}
		}
		else
		{
			uncle = Node->Parent->Parent->Left;

			if (uncle->Color == TColor::Red)
			{
				Node->Parent->Color = TColor::Black;
				uncle->Color = TColor::Black;

				Node->Parent->Parent->Color = TColor::Red;
				Node = Node->Parent->Parent;
			}
			else
			{
				if (Node == Node->Parent->Left)
				{
					Node = Node->Parent;
					OBTreeUtils::RotateRight(Node);
				}
				Node->Parent->Color = TColor::Black;
				Node->Parent->Parent->Color = TColor::Red;
				OBTreeUtils::RotateLeft(Node->Parent->Parent);
			}
		}
	}
	Root->Color = TColor::Black;
}

template<typename KeyType, typename ValueType>
void ORedBlackTree<KeyType, ValueType>::Insert(KeyType Key, ValueType Value)
{
	auto currentNode = Root;
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
	auto newNode = NodeType(Key, Value, NullNode);
	newNode.Parent = parent;

	if (parent == NullNode)
	{
		Root = newNode;
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
bool ORedBlackTree<KeyType, ValueType>::DoesNodeExist(ORedBlackTree::NodeType* Node)
{
	Node != NullNode;
}

} // namespace RAPI