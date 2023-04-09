#pragma once

#include "Threads/Thread.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"

#include <boost/thread/exceptions.hpp>

namespace RAPI
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
		else if (ThreadConditionAny)
		{
			ThreadConditionAny->notify_all();
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

	template<typename TLockable>
	void Wait(OConditionVariableAny& CV, TLockable& Lock);

	OInterruptFlag() = default;

private:
	OAtomic<bool> Flag;
	OConditionVariable* ThreadCondition{ nullptr };
	OConditionVariableAny* ThreadConditionAny{ nullptr };
	OMutex SetClearMutex;
};

static thread_local OInterruptFlag LocalThreadInterruptFlag; // NOLINT
FORCEINLINE void InterruptionPoint()
{
	if (LocalThreadInterruptFlag.IsSet())
	{
		throw boost::thread_interrupted();
	}
}

template<typename TLockable>
void OInterruptFlag::Wait(OConditionVariableAny& CV, TLockable& Lock)
{
	{
		struct SCustomLock
		{
			OInterruptFlag* Self;
			TLockable& LK;
			SCustomLock(OInterruptFlag* SelfFlag, OConditionVariableAny& Condition, TLockable& Lock)
			    : Self(SelfFlag), LK(Lock)
			{
				Self->SetClearMutex.lock();
				Self->ThreadConditionAny = &Condition;
			}
			void Unlock()
			{
				LK.unlock();
				Self->SetClearMutex.unlock();
			}

			void Lock()
			{
				std::lock(Self->SetClearMutex, LK);
			}

			~SCustomLock()
			{
				Self->ThreadConditionAny = nullptr;
				Self->SetClearMutex.unlock();
			}
		};

		SCustomLock cL(this, CV, Lock);
		InterruptionPoint();
		CV.wait(cL);
		InterruptionPoint();
	}
}
} // namespace RAPI