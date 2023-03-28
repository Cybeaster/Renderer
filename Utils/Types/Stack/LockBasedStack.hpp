#pragma once

#ifndef RENDERAPI_LOCKBASEDSTACK_HPP
#define RENDERAPI_LOCKBASEDSTACK_HPP
#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"

#include <stack>

namespace RenderAPI
{

/**
 * @brief Very slow implementation. Doesn't use whole concurrency features.
 */
template<typename T>
class OLockBasedStack
{
public:
	OLockBasedStack() = default;
	OLockBasedStack& operator=(const OLockBasedStack& Other) = delete;

	OLockBasedStack(const OLockBasedStack& Other)
	{
		OMutexGuard lock(Other.Mutex);
		Data = Other.Data;
	}

	template<typename Type>
	void Push(Type&& Value)
	{
		OMutexGuard lock(Mutex);
		Data.push(Forward(Value));
	}

	OSharedPtr<T> Pop()
	{
		OMutexGuard lock(Mutex);
		if (ENSURE(!Data.empty()))
		{
			OSharedPtr<T> result(MakeShared<T>(Move(Data.top())));
			Data.pop();
			return result;
		}
		return {};
	}

	void Pop(T& OutResult)
	{
		OMutexGuard lock(Mutex);
		if (ENSURE(!Data.empty()))
		{
			OutResult = Move(Data.top());
			Data.pop();
		}
	}

	NODISCARD bool IsEmpty()
	{
		OMutexGuard lock(Mutex);
		return Data.empty();
	}

private:
	std::stack<T> Data;
	OMutex Mutex;
};
} // namespace RenderAPI
#endif // RENDERAPI_LOCKBASEDSTACK_HPP
