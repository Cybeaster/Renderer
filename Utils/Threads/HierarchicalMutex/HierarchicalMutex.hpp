#pragma once
#include <Thread.hpp>
#include <Types.hpp>

namespace RenderAPI
{
namespace Thread
{
class HierarchicalMutex
{
public:
	explicit HierarchicalMutex(uint64 Value)
	    : HierarchyValue(Value), PreviousHierarchyValue(0) {}

	void Lock();
	void Unlock();
	bool TryLock();

	HierarchicalMutex(/* args */) = delete;
	~HierarchicalMutex() noexcept;

private:
	bool CheckForHierarchyViolation();
	void UpdateHierarchyValue();

	uint64 HierarchyValue;
	uint64 PreviousHierarchyValue;
	static thread_local uint64 ThreadHierarchyValue;
	OMutex InternalMutex;
};
} // namespace Thread

} // namespace RenderAPI
