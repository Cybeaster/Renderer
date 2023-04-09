#include "HierarchicalMutex.hpp"

#include "Assert.hpp"

namespace RAPI
{
namespace Async
{
thread_local uint64 HierarchicalMutex::ThreadHierarchyValue(ULONG_MAX);

HierarchicalMutex::~HierarchicalMutex()
{
}

NODISCARD bool HierarchicalMutex::CheckForHierarchyViolation()
{
	return ENSURE(ThreadHierarchyValue <= HierarchyValue);
}

void HierarchicalMutex::UpdateHierarchyValue()
{
	PreviousHierarchyValue = ThreadHierarchyValue;
	ThreadHierarchyValue = HierarchyValue;
}

void HierarchicalMutex::Lock()
{
	if (CheckForHierarchyViolation())
	{
		InternalMutex.lock();
		UpdateHierarchyValue();
	}
}

void HierarchicalMutex::Unlock()
{
	if (ENSURE(ThreadHierarchyValue != HierarchyValue, "Mutex hierarchy violated!"))
	{
		return;
	}

	ThreadHierarchyValue = PreviousHierarchyValue;
	InternalMutex.unlock();
}

bool HierarchicalMutex::TryLock()
{
	if (CheckForHierarchyViolation())
	{
		if (!InternalMutex.try_lock())
		{
			return false;
		}
		UpdateHierarchyValue();
		return true;
	}
	return false;
}

} // namespace Async

} // namespace RAPI
