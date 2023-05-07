#pragma once

#include "Logging/Printer.hpp"
#include "Math.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{
enum class ETraverseType
{
	Symmetric, // left child -> parent -> right child
	Reverse,
	Direct
};

class OBTreeUtils
{
public:
	template<typename NodeType, typename Operation>
	static void TraverseTreeWith(NodeType* Where, ETraverseType Type, Operation Op);

	template<typename NodeType, typename Operation>
	static void Print(NodeType* Where, ETraverseType Type, OPrinter* Printer);

	template<typename NodeType, typename KeyType, typename ValueType>
	static ValueType FindValue(NodeType* Node, KeyType Key);

	template<typename NodeType, typename KeyType>
	static NodeType* FindNode(NodeType* Node, KeyType);

	template<typename NodeType, typename KeyType, typename ValueType>
	static void Insert(NodeType* Where, KeyType Key, ValueType Value);

	template<typename NodeType>
	static NodeType* GetMax(NodeType* Where);

	template<typename NodeType>
	static NodeType* GetMin(NodeType* Where);

	template<typename NodeType>
	static void RotateRight(NodeType* Node);

	template<typename NodeType>
	static void RotateLeft(NodeType* Node);

	template<typename NodeType>
	static auto GetHeight(NodeType* Node);

	template<typename NodeType>
	static void IncrementHeight(NodeType* Node);

	template<typename NodeType>
	static int8 GetBalance(NodeType* Node);

	template<typename NodeType>
	static void Swap(NodeType* First, NodeType* Second);

private:
	template<typename NodeType, typename KeyType, typename ValueType>
	static void Insert(NodeType* Where, NodeType* Node);
};

template<typename NodeType, typename Operation>
void OBTreeUtils::Print(NodeType* Where, ETraverseType Type, OPrinter* Printer)
{
	TraverseTreeWith(Where, Type, [Printer](NodeType* Node)
	                 { *Printer << "Key: " << Node->Key << "Value: " << Node->Value << "\n"; });
}


template<typename NodeType, typename KeyType>
NodeType* OBTreeUtils::FindNode(NodeType* Node, KeyType Key)
{
	if(Node != nullptr)
	{
		if(Node->Key == Key)
		{
			return Node;
		}
		else
		{
			return Node->Key > Key ? FindNode(Node->Left,Key) : FindNode(Node->Right, Key);
		}
	}
	return nullptr;
}

template<typename NodeType, typename Operation>
void OBTreeUtils::TraverseTreeWith(NodeType* Where, ETraverseType Type, Operation Op)
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

template<typename NodeType>
auto OBTreeUtils::GetHeight(NodeType* Node)
{
	return Node == nullptr ? -1 : Node->Height;
}

template<typename NodeType>
void OBTreeUtils::IncrementHeight(NodeType* Node)
{
	Node->Height = SMath::Max(GetHeight(Node->Left), GetHeight(Node->Right)) + 1;
}

template<typename NodeType>
int8 OBTreeUtils::GetBalance(NodeType* Node)
{
	return Node == nullptr ? 0 : GetHeight(Node->Right) - GetHeight(Node->Left);
}

template<typename NodeType>
void OBTreeUtils::Swap(NodeType* First, NodeType* Second)
{
	const auto firstKey = First->Key;
	const auto firstValue = First->Value;

	First->Value = Move(Second->Value);
	First->Key = Move(Second->Key);

	Second->Key = Move(firstKey);
	Second->Value = Move(firstValue);
}

template<typename NodeType>
void OBTreeUtils::RotateLeft(NodeType* Node)
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

template<typename NodeType>
void OBTreeUtils::RotateRight(NodeType* Node)
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

template<typename NodeType>
NodeType* OBTreeUtils::GetMin(NodeType* Where)
{
	if (Where->Left == nullptr)
	{
		return Where->Left;
	}

	return GetMin(Where->Left);
}

template<typename NodeType>
NodeType* OBTreeUtils::GetMax(NodeType* Where)
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

template<typename NodeType, typename KeyType, typename ValueType>
void OBTreeUtils::Insert(NodeType* Where, NodeType* Node)
{
	if (Node->Key > Where->Key)
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
	Insert(Where, Node);
}

template<typename NodeType, typename KeyType, typename ValueType>
void OBTreeUtils::Insert(NodeType* Where, KeyType Key, ValueType Value)
{
	auto newNode = new NodeType(Key, Value);
	if (Where == nullptr)
	{
		Where = newNode;
	}
	else
	{
		Insert(Where, newNode);
	}
}

template<typename NodeType, typename KeyType, typename ValueType>
ValueType OBTreeUtils::FindValue(NodeType* Node, KeyType Key)
{
	if (ENSURE(Node != nullptr))
	{
		if (Node->Key == Key)
		{
			return Node->Value;
		}

		return Node->Key > Key ? FindValue(Node->Left, Key) : FindValue(Node->Right, Key);
	}
	return ValueType();
}

template<typename KeyType, typename ValueType, typename NodeType>
class OBSearchTreeBase
{
public:
	ValueType Find(KeyType Key)
	{
		OBTreeUtils::FindValue(Root, Key);
	}

	ValueType GetMax()
	{
		if (ENSURE(Root != nullptr))
		{
			OBTreeUtils::GetMax(Root);
		}
		return ValueType();
	}

	ValueType GetMin()
	{
		if (ENSURE(Root != nullptr))
		{
			OBTreeUtils::GetMin(Root);
		}
		return ValueType();
	}

	void Print(OPrinter* Printer, ETraverseType TraversType = ETraverseType::Direct)
	{
		OBTreeUtils::Print(Root, ETraverseType::Direct);
	}

	virtual void Insert(KeyType Key, ValueType Value) = 0;
	virtual bool Remove(KeyType Key) = 0;

protected:
	OUniquePtr<NodeType> Root;
};

} // namespace RAPI