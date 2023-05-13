#pragma once

#include "Array.hpp"
#include "HashMap/Hash.hpp"
#include "List.hpp"
#include "Printable.hpp"
#include "Sets/MultiIndex.hpp"
#include "Sets/Set.hpp"
#include "SmartPtr.hpp"
#include "Stack/Stack.hpp"
#include "Types.hpp"
#include "Utils/Types/Sets/Set.hpp"
#include "Vector.hpp"
namespace RAPI::Graph
{
template<typename ValueType>
struct SEdge;

template<typename ValueType>
using TEdge = OSharedPtr<SEdge<ValueType>>;

template<typename ValueType>
struct SNode
{
	ValueType Value;
	OSet<TEdge<ValueType>> Edges;
	OHashMap<OWeakPtr<SNode>, TEdge<ValueType>> Parents;
	explicit SNode(ValueType Other)
	    : Value(Other) {}
};

template<typename ValueType>
struct SEdge
{
	OWeakPtr<SNode<ValueType>> AdjacentNode;

	OSharedPtr<SNode<ValueType>> GetAdjacentNode()
	{
		return AdjacentNode.lock();
	}

	int32 Weight;
	SEdge(SNode<ValueType>* Node, int32 Other)
	    : AdjacentNode(Node), Weight(Other) {}
};

template<typename ValueType>
class OGraph : public IPrintable
{
	using NodeType = SNode<ValueType>*;
	using TNode = OSharedPtr<NodeType>;
	using TEdge = SEdge<ValueType>;

	struct SPathNode
	{
		NodeType Node;
		OSharedPtr<SPathNode> Parent;
		SPathNode(NodeType Other, OSharedPtr<SPathNode> OtherParent)
		    : Node(Other), Parent(OtherParent) {}
	};

public:
	TNode AddOrGetNode(ValueType Value);
	/**@param Data Matrix N X 3 (N rows and 3 columns) */
	void Construct(OVector<OArray<ValueType, 3>> Data);

	void Print(OPrinter* Printer) override;
	void Print(TNode Node, OPrinter* Printer);

	template<typename Operation>
	void DFSWithOperation(Operation&& Op, bool Recursive);

	template<typename Operation>
	void BFSWithOperation(Operation&& Op, bool Recursive);

	OSharedPtr<OList<OSet<NodeType>>> FindAllPaths(NodeType Start, NodeType End);

private:
	bool FindPath(NodeType Start, NodeType End, OSet<NodeType>& Passed, OList<NodeType>& Path);
	void FindAllPaths(NodeType Start, NodeType End, OSet<NodeType>& Passed, OList<OSet<NodeType>>& Paths);

	template<typename Operation>
	void DFS(NodeType Node, OSet<NodeType>& Passed, Operation&& Op);

	template<typename Operation>
	void NonRecursiveDFS(NodeType Node, OSet<NodeType>& Passed, Operation&& Op);

	template<typename Operation>
	void NonRecursiveBFS(NodeType Node, OSet<NodeType> VisitingOrPassed, Operation&& Op);

	OList<NodeType> ExtractPath(OSharedPtr<SPathNode> Node);

	OSharedPtr<SPathNode> FindShortestUnweightedPath(NodeType Start, NodeType End);

	// Weighted search
	NodeType GetNodeWithMinTimeToIt(OSet<NodeType>& UnprocessedNodes, OHashMap<NodeType, int32>& TimeToNodes);
	void CalcTimeToEachNode(OSet<NodeType>& UnprocessedNodes, OHashMap<NodeType, int32>& TimeToNodes);

	OList<NodeType> FindShortestWeightedPath(NodeType Start, NodeType End);
	// Weighted search

	OHashMap<ValueType, TNode> Graph;
};

template<typename ValueType>
void OGraph<ValueType>::CalcTimeToEachNode(OSet<NodeType>& UnprocessedNodes, OHashMap<NodeType, int32>& TimeToNodes)
{
	while (!UnprocessedNodes.empty())
	{
		auto node = GetNodeWithMinTimeToIt(UnprocessedNodes, TimeToNodes);
		if (TimeToNodes[node] == STypeLimits<int32>::Max())
		{
			return;
		}

		for (auto edge : node->Edges)
		{
			auto adjacent = edge->GetAdjacentNode();
			if (UnprocessedNodes.contains(adjacent))
			{
				auto time = TimeToNodes[node] + edge->Weight;
				if (TimeToNodes[adjacent] > time)
				{
					TimeToNodes[adjacent] = time;
				}
			}
		}
		std::remove(UnprocessedNodes.begin(), UnprocessedNodes.end(), node);
	}
}

template<typename ValueType>
OGraph<ValueType>::NodeType OGraph<ValueType>::GetNodeWithMinTimeToIt(OSet<NodeType>& UnprocessedNodes, OHashMap<OGraph::NodeType, int32>& TimeToNodes)
{
	NodeType nodeWithMinTime = nullptr;
	auto minTime = STypeLimits<int32>::Max();
	for (auto node : UnprocessedNodes)
	{
		auto time = TimeToNodes[node];
		if (time < minTime)
		{
			minTime = time;
			nodeWithMinTime = node;
		}
	}
	return nodeWithMinTime;
}

template<typename ValueType>
OList<OGraph<ValueType>::NodeType> OGraph<ValueType>::FindShortestWeightedPath(OGraph::NodeType Start, OGraph::NodeType End)
{
	OHashMap<NodeType, int32> timeToNodes;
	OSet<NodeType> unprocessedNodes;

	for (auto entry : Graph)
	{
		auto node = entry.second;
		unprocessedNodes.insert(node);
		timeToNodes.insert({ node, STypeLimits<int32>::Max() });
	}
	timeToNodes[Start] = 0;
	CalcTimeToEachNode(unprocessedNodes, timeToNodes);

	if (timeToNodes[End] == STypeLimits<int32>::Max())
	{
		return {};
	}
	
	OList<NodeType> result;
	NodeType current = End;

	while (current != Start)
	{
		auto minTime = timeToNodes[End];
		result.push_front(current);

		for (auto& entry : current->Parents)
		{
			auto parent = entry->first.lock().get();
			auto parentEdge = entry->second;

			if (timeToNodes.contains(parent))
			{
				continue;
			}

			bool prevNodeFound = timeToNodes[parent] + parentEdge->Weight == minTime;
			if (prevNodeFound)
			{
				timeToNodes.erase(timeToNodes.begin(), std::remove(timeToNodes.begin(), timeToNodes.end(), current));
				current = parent;
				break;
			}
		}
	}
	result.push_front(Start);
}

template<typename ValueType>
OList<OGraph<ValueType>::NodeType> OGraph<ValueType>::ExtractPath(OSharedPtr<SPathNode> Node)
{
	OList<NodeType> path;
	while (Node != nullptr)
	{
		path.push_front(Node->Node);
		Node = Node->Parent;
	}
	return Move(path);
}

template<typename ValueType>
typename OSharedPtr<OGraph<ValueType>::SPathNode> OGraph<ValueType>::FindShortestUnweightedPath(NodeType Start, NodeType End)
{
	OSet<NodeType> visitingOrPassed;
	OList<OSharedPtr<SPathNode>> queue;
	queue.push_back(Start);
	while (!queue.empty())
	{
		auto pathNode = queue.front();
		queue.pop_front();
		if (pathNode->Node == End)
		{
			return pathNode;
		}

		for (auto edge : pathNode->Node->Edges)
		{
			auto adjacentNode = edge->GetAdjacentNode();
			if (visitingOrPassed.contains(adjacentNode))
			{
				continue;
			}
			if (adjacentNode == End)
			{
				return MakeShared<SPathNode>(adjacentNode, pathNode);
			}
			queue.push_back(MakeShared<SPathNode>(adjacentNode, pathNode));
			visitingOrPassed.insert(pathNode->Node);
		}
	}
}

template<typename ValueType>
template<typename Operation>
void OGraph<ValueType>::BFSWithOperation(Operation&& Op, bool Recursive)
{
	OSet<NodeType> passed;
	for (auto& entry : Graph)
	{
		auto node = entry.second.get();
		if (Recursive)
		{
		}
		else
		{
			NonRecursiveBFS(node, passed, Op);
		}
	}
}

template<typename ValueType>
template<typename Operation>
void OGraph<ValueType>::NonRecursiveBFS(OGraph::NodeType Node, OSet<NodeType> VisitingOrPassed, Operation&& Op)
{
	VisitingOrPassed.insert(Node);
	OList<NodeType> queue;
	queue.push_back(Node);
	while (!queue.empty())
	{
		auto first = queue.front();
		queue.pop_front();
		Op(first);
		for (auto edge : first->Edges)
		{
			if (!VisitingOrPassed.contains(edge->GetAdjacentNode()))
			{
				queue.push_back(edge->GetAdjacentNode());
			}
		}
	}
}

template<typename ValueType>
OSharedPtr<OList<OSet<OGraph<ValueType>::NodeType>>> OGraph<ValueType>::FindAllPaths(NodeType Start, NodeType End)
{
	auto result = MakeShared<OList<OSet<ValueType>>>();
	OSet<NodeType> passed;
	if (Start == nullptr)
	{
		return result;
	}
	FindAllPaths(Start, End, passed, result);
	return result;
}

template<typename ValueType>
void OGraph<ValueType>::FindAllPaths(NodeType Start, NodeType End, OSet<NodeType>& Passed, OList<OSet<NodeType>>& Paths)
{
	if (Start == End)
	{
		Paths.push_back(OSet<ValueType>(Passed));
		auto last = std::find_end(Paths.begin(), Paths.end());
		last->insert(End);
	}

	Passed.insert(Start);
	for (auto edge : Start->Edges)
	{
		if (!Passed.contains(edge->GetAdjacentNode()))
		{
			FindAllPaths(edge->GetAdjacentNode(), End, Passed, Paths);
		}
	}
	std::remove(Passed.begin(), Passed.end(), Start);
}

template<typename ValueType>
bool OGraph<ValueType>::FindPath(NodeType Start, NodeType End, OSet<NodeType>& Passed, OList<NodeType>& Path)
{
	Passed.insert(Start);
	if (Start == End)
	{
		Path.push_front(Start);
		return true;
	}

	for (auto& edge : Start->Edges)
	{
		if (!Passed.contains(edge->GetAdjacentNode()))
		{
			if (FindPath(edge->GetAdjacentNode(), End, Passed, Path))
			{
				Path.push_front(Start);
				return true;
			}
		}
	}
}

template<typename ValueType>
template<typename Operation>
void OGraph<ValueType>::NonRecursiveDFS(NodeType Node, OSet<NodeType>& Passed, Operation&& Op)
{
	OStack<NodeType> stack;
	stack.push(Node);

	while (stack.size() != 0)
	{
		auto node = stack.top();
		stack.pop();

		if (!Passed.contains(node))
		{
			Passed.insert(node);
			Op(node);
		}
		bool hasChildren = false;

		for (auto edge : node.Edges)
		{
			if (Passed.contains(edge->GetAdjacentNode()))
			{
				stack.push(edge->GetAdjacentNode());
				hasChildren = true;
				break;
			}
		}
		if (!hasChildren)
		{
			stack.pop();
		}
	}
}

template<typename ValueType>
template<typename Operation>
void OGraph<ValueType>::DFSWithOperation(Operation&& Op, bool Recursive)
{
	OSet<NodeType> passed;
	for (auto entry : Graph)
	{
		auto node = entry.second.get();
		if (!passed.contains(node))
		{
			if (Recursive)
			{
				DFS(node, passed, Op);
			}
			else
			{
				NonRecursiveDFS(node, passed, Op);
			}
		}
	}
}

template<typename ValueType>
void OGraph<ValueType>::Print(OGraph::TNode Node, OPrinter* Printer)
{
	Printer << Node.get()->Value;
}

template<typename ValueType>
template<typename Operation>
void OGraph<ValueType>::DFS(SNode<ValueType> Node, OSet<NodeType>& Passed, Operation&& Op)
{
	Passed.insert(Node);
	Op(Node);
	for (auto& edge : Node->Edges)
	{
		if (!Passed.contains(edge->GetAdjacentNode()))
		{
			DFS(edge->GetAdjacentNode(), Passed);
		}
	}
}

template<typename ValueType>
void OGraph<ValueType>::Print(OPrinter* Printer)
{
}

template<typename ValueType>
void OGraph<ValueType>::Construct(OVector<OArray<ValueType, 3>> Data)
{
	for (auto& row : Data)
	{
		auto node = AddOrGetNode(row[0]);
		auto adjacentNode = AddOrGetNode(row[1]);
		if (adjacentNode == nullptr)
		{
			continue;
		}

		auto edge = MakeShared<TEdge>(adjacentNode, row[2]);
		node->Edges.insert(edge);
		adjacentNode->Parents.insert(node, edge);
	}
}

template<typename ValueType>
OGraph<ValueType>::TNode OGraph<ValueType>::AddOrGetNode(ValueType Value)
{
	if (Value == -1)
	{
		return nullptr;
	}
	if (Graph.contains(Value))
	{
		return Graph[Value];
	}

	auto newNode = MakeShared<TNode>(Value);
	Graph[Value] = Move(newNode);
	return newNode;
}

} // namespace RAPI::Graph