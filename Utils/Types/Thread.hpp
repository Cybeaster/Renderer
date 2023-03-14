#pragma once
#include "../Utils/Threads/JoiningThread.hpp"

#include <atomic>
#include <future>
#include <mutex>

using OMutex = std::mutex;

using OMutexGuard = std::lock_guard<OMutex>;
using OUniqueLock = std::unique_lock<OMutex>;

template<typename T>
using OFuture = std::future<T>;

template<typename T>
using OSharedFuture = std::shared_future<T>;

using OConditionVariable = std::condition_variable;

template<typename T>
using OAtomic = std::atomic<T>;

using OAtomicFlag = std::atomic_flag;

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
