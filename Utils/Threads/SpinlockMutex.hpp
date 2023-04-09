#pragma once

#include "Threads/Thread.hpp"
namespace RAPI
{
class OSpinlockMutex
{
public:
	OSpinlockMutex() = default;

	void Lock()
	{
		while (Flag.test_and_set(EMemoryOrder::Acquire))
			;
	}

	void Unlock()
	{
		Flag.clear(std::memory_order_release);
	}

private:
	OAtomicFlag Flag;
};

} // namespace RAPI