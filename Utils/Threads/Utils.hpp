#pragma once

#include "Threads/InterruptibleThread/InterruptFlag.hpp"
#include "Time.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Utils/Types/Threads/Thread.hpp"
#include "Vector.hpp"

#include <boost/thread/exceptions.hpp>

namespace RenderAPI
{

class OJoinThreads
{
public:
	explicit OJoinThreads(OVector<OThread>& Other)
	    : Threads(Other) {}

	~OJoinThreads()
	{
		for (auto& thread : Threads)
		{
			if (thread.joinable())
			{
				thread.join();
			}
		}
	}

private:
	OVector<OThread>& Threads;
};

class OSpinlockMutex
{
public:
	OSpinlockMutex() = default;

	void Lock()
	{
		while (Flag.test_and_set(std::memory_order_acquire))
			;
	}

	void Unlock()
	{
		Flag.clear(std::memory_order_release);
	}

private:
	OAtomicFlag Flag;
};

class OThreadUtils
{
public:
	static thread_local OInterruptFlag LocalThreadInterruptFlag; // NOLINT

	NODISCARD static OString GetFormattedThreadID();

	FORCEINLINE static void InterruptionPoint()
	{
		if (LocalThreadInterruptFlag.IsSet())
		{
			throw boost::thread_interrupted();
		}
	}

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
};

} // namespace RenderAPI
