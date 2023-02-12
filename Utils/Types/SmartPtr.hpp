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

template<typename T>
FORCEINLINE auto MakeShared(T* Arg)
{
	return std::make_shared<T>(Arg);
}

template<typename T>
using OTWeakPtr = std::weak_ptr<T>;

template<typename T>
using OTSharedPtr = std::shared_ptr<T>;

template<typename T>
using OTUniquePtr = std::unique_ptr<T>;

} // namespace RenderAPI
