#pragma once
#include "../Utils/Threads/JoiningThread.hpp"

#include <atomic>
#include <future>
#include <mutex>
#include <shared_mutex>

using OMutex = std::mutex;
using OSharedMutex = std::shared_mutex;
using OMutexGuard = std::lock_guard<OMutex>;

using OSharedMutexLock = std::shared_lock<OSharedMutex>;
using OUniqueMutexLock = std::unique_lock<OMutex>;

template<typename T>
using OUniqueLock = std::unique_lock<T>;

template<typename T>
using OSharedLock = std::shared_lock<T>;

template<typename T>
using OFuture = std::future<T>;

template<typename T>
using OSharedFuture = std::shared_future<T>;

using OConditionVariable = std::condition_variable;

template<typename T>
using OAtomic = std::atomic<T>;

using ThreadID = std::thread::id;

inline decltype(auto) GetThisThreadID() noexcept
{
	return std::this_thread::get_id();
}

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
