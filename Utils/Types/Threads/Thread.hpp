#pragma once
#include "Utils/Threads/JoiningThread.hpp"

#include <atomic>
#include <future>
#include <mutex>
#include <shared_mutex>

using OMutex = std::mutex;
using OSharedMutex = std::shared_mutex;
using OMutexGuard = std::lock_guard<OMutex>;
using OSharedMutexLock = std::shared_lock<OSharedMutex>;
using OUniqueMutexLock = std::unique_lock<OMutex>;

using OThread = std::thread;

template<typename T>
using OUniqueLock = std::unique_lock<T>;

template<typename T>
using OSharedLock = std::shared_lock<T>;

template<typename T>
using OFuture = std::future<T>;

template<typename... Args>
using OPackagedTask = std::packaged_task<Args...>;

template<typename T>
using OSharedFuture = std::shared_future<T>;

template<typename T>
using OPromise = std::promise<T>;

using OConditionVariable = std::condition_variable;
using OConditionVariableAny = std::condition_variable_any;

template<typename T>
using OAtomic = std::atomic<T>;

using ThreadID = std::thread::id;

namespace NThisThread
{
inline decltype(auto) GetThreadID() noexcept
{
	return std::this_thread::get_id();
}

inline void Yield() noexcept
{
	std::this_thread::yield();
}

} // namespace NThisThread

using OAtomicFlag = std::atomic_flag;

namespace EFutureStatus
{
constexpr auto Ready = std::future_status::ready;
constexpr auto TimeOut = std::future_status::timeout;
constexpr auto Deffered = std::future_status::deferred;
}; // namespace EFutureStatus

namespace EMemoryOrder
{
constexpr auto Relaxed = std::memory_order::relaxed;
constexpr auto Consume = std::memory_order::consume;
constexpr auto Acquire = std::memory_order::acquire;
constexpr auto Release = std::memory_order::release;
constexpr auto AcqRel = std::memory_order::acq_rel;
constexpr auto SeqCst = std::memory_order::seq_cst;
} // namespace EMemoryOrder
