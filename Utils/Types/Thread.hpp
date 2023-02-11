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

using OConditionVariable = std::condition_variable;

template<typename T>
using OAtomic = std::atomic<T>;
