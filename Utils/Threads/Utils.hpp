#pragma once

#include "JoinThreads.hpp"
#include "SpinlockMutex.hpp"
#include "Threads/InterruptibleThread/InterruptFlag.hpp"
#include "Time.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils/Types/Threads/Thread.hpp"
#include "Vector.hpp"

#include <boost/thread/exceptions.hpp>

namespace RAPI
{

class OThreadUtils
{
public:
	NODISCARD static OString GetFormattedThreadID();

	FORCEINLINE static void InterruptibleWait(OConditionVariable& CV, OUniqueLock<OMutex>& Lock)
	{
		InterruptionPoint();

		LocalThreadInterruptFlag.SetConditionVariable(CV);
		OInterruptFlag::SClearCondVariableOnDestruct guard;
		InterruptionPoint();
		CV.wait_for(Lock, SMillSeconds(1));
		LocalThreadInterruptFlag.ClearConditionVariable();
		InterruptionPoint();
	}

	template<typename TPredicate>
	FORCEINLINE static void InterruptibleWait(OConditionVariable& CV, OUniqueLock<OMutex>& Lock, TPredicate Predicate)
	{
		InterruptionPoint();
		LocalThreadInterruptFlag.SetConditionVariable(CV);
		OInterruptFlag::SClearCondVariableOnDestruct guard;

		while (!LocalThreadInterruptFlag.IsSet() && !Predicate())
		{
			CV.wait_for(Lock, SMillSeconds(1));
		}
		InterruptionPoint();
	}

	template<typename T>
	void InterruptibleWait(OFuture<T>& Future)
	{
		OMutex lk;
		while (!LocalThreadInterruptFlag.IsSet())
		{
			if (Future.wait_for(lk, SMillSeconds(1)) == EFutureStatus::Ready)
			{
				break;
			}
		}
		InterruptionPoint();
	}

	template<typename Lockable>
	void InterruptibleWait(OConditionVariableAny& CV, Lockable& LK)
	{
		LocalThreadInterruptFlag.Wait(CV, LK);
	}
};

} // namespace RAPI
