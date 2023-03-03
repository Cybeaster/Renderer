#pragma once
#include "Types.hpp"

#include <memory>

namespace RenderAPI
{

template<typename T>
FORCEINLINE auto MakeUnique(T* Arg)
{
	return std::make_unique<T>(Arg);
}

template<class Type, typename... Payload>
FORCEINLINE auto MakeShared(Payload&&... Args)
{
	return std::make_shared<Type>(Args...);
}

template<class Type>
FORCEINLINE auto MakeShared(Type* Arg)
{
	return std::make_shared<Type>(Arg);
}

template<typename T>
using OTWeakPtr = std::weak_ptr<T>;

template<typename T>
using OTSharedPtr = std::shared_ptr<T>;

template<typename T>
using OTUniquePtr = std::unique_ptr<T>;

} // namespace RenderAPI
