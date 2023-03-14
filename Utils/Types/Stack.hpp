#pragma once

#ifndef RENDERAPI_STACK_HPP
#define RENDERAPI_STACK_HPP
#include "SmartPtr.hpp"
#include "Thread.hpp"

#include <stack>

namespace RenderAPI
{

template<typename T>
class OThreadSafeStack
{
public:
	OThreadSafeStack() = default;
	OThreadSafeStack& operator=(const OThreadSafeStack& Other) = delete;

	OThreadSafeStack(const OThreadSafeStack& Other)
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
		if (Data.empty())
		{
		}

		OSharedPtr<T> result(MakeShared<T>(Move(Data.top())));
		Data.pop();
		return result;
	}

	void Pop(T& OutResult)
	{
		OMutexGuard lock(Mutex);
		if (ENSURE(Data.empty()))
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
#endif // RENDERAPI_STACK_HPP
