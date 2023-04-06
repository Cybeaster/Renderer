#pragma once

#include "Threads/Thread.hpp"
#include "TypeTraits.hpp"

namespace RenderAPI
{

class OInterruptFlag
{
public:
	struct SClearCondVariableOnDestruct
	{
		~SClearCondVariableOnDestruct();
	};

	void Set()
	{
		Flag.store(true, EMemoryOrder::Relaxed);
		OMutexGuard lock(SetClearMutex);
		if (ThreadCondition)
		{
			ThreadCondition->notify_all();
		}
	}
	NODISCARD bool IsSet() const
	{
		return Flag.load(EMemoryOrder::Relaxed);
	}

	void SetConditionVariable(OConditionVariable& CV)
	{
		OMutexGuard guard(SetClearMutex);
		ThreadCondition = &CV;
	}

	void ClearConditionVariable()
	{
		OMutexGuard guard(SetClearMutex);
		ThreadCondition = nullptr;
	}

	OInterruptFlag() = default;

private:
	OAtomic<bool> Flag;
	OConditionVariable* ThreadCondition{ nullptr };
	OMutex SetClearMutex;
};

} // namespace RenderAPI