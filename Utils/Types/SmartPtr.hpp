#pragma once
#include "Types.hpp"
#include "boost/make_shared.hpp"
#include "boost/shared_ptr.hpp"

#include <memory>

namespace RAPI
{

template<typename T>
FORCEINLINE auto MakeUnique(T* Arg)
{
	return std::make_unique<T>(Arg);
}

template<class Type, typename... Payload>
FORCEINLINE auto MakeShared(Payload&&... Args)
{
	return boost::make_shared<Type>(Args...);
}

template<class Type>
FORCEINLINE auto MakeShared(Type* Arg)
{
	return std::make_shared<Type>(Arg);
}

template<typename T>
using OWeakPtr = std::weak_ptr<T>;

template<typename T>
using OSharedPtr = boost::shared_ptr<T>;

template<typename T>
using OUniquePtr = std::unique_ptr<T>;

} // namespace RAPI
