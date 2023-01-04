#pragma once
#include "../Utils/Threads/JoiningThread.hpp"
#include <mutex>
#include <atomic>
#include <future>

using TMutex = std::mutex;

using TMutexGuard = std::lock_guard<TMutex>;
using TUniqueLock = std::unique_lock<TMutex>;

template <typename T>
using TTFuture = std::future<T>;

using TConditionVariable = std::condition_variable;

template <typename T>
using TTAtomic = std::atomic<T>;
