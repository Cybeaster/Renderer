#pragma once

#include "HashMap/Hash.hpp"
#include "List.hpp"
#include "Sets/MultiIndex.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"
#include "Utils/Types/Sets/Set.hpp"
namespace RAPI::Graph
{
template<typename ValueType>
struct SEdge;

template<typename ValueType>
struct SNode
{
	ValueType Value;
	LinkedHashSet<ValueType> Edges;
	OHashMap<SNode, SEdge<ValueType>> Parents;
	explicit SNode(ValueType Other)
	    : Value(Other) {}
};

template<typename ValueType>
struct SEdge
{
	SNode<ValueType>* AdjacentNode;
	int32 Weight;
	SEdge(SNode<ValueType>* Node, int32 Other)
	    : AdjacentNode(Node), Weight(Other) {}
};

template<typename KeyType, typename ValueType>
class OGraph
{
	using TNode = SNode<ValueType>;

public:
	TNode* AddOrGetNode(KeyType Key);

	OHashMap<KeyType, OSharedPtr<TNode>> Graph;
};

template<typename KeyType, typename ValueType>
OGraph<KeyType, ValueType>::TNode* OGraph<KeyType, ValueType>::AddOrGetNode(KeyType Key)
{
	if (Key == -1)
	{
		return nullptr;
	}
	if (Graph.contains(Key))
	{
		return Graph[Key];
	}

	auto newNode = MakeShared<TNode>(Key);
	Graph[Key] = Move(newNode);
	return newNode;
}

} // namespace RAPI::Graph