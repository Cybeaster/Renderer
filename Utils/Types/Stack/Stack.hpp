#pragma once
#include "LockBasedStack.hpp"
#include "LockFreeStack.hpp"

#include <stack>

template<typename T>
using OStack = std::stack<T>;

template<typename T>
using OThreadSafeStack = RAPI::OLockBasedStack<T>;