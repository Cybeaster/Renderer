#pragma once
#include "Types.hpp"
#include "boost/make_shared.hpp"
#include "boost/shared_ptr.hpp"

#include <memory>

namespace RAPI
{

template<typename T, typename... Payload>
FORCEINLINE auto MakeUnique(Payload&&... Args)
{
	return std::make_unique<T>(Args...);
}

template<class Type, typename... Payload>
FORCEINLINE auto MakeShared(Payload&&... Args)
{
	return boost::make_shared<Type>(Args...);
}

template<class Type>
FORCEINLINE auto MakeShared(Type* Arg)
{
	return boost::make_shared<Type>(Arg);
}

template<typename T>
using OWeakPtr = std::weak_ptr<T>;

template<typename T>
using OSharedPtr = boost::shared_ptr<T>;

template<typename T>
using OUniquePtr = std::unique_ptr<T>;

} // namespace RAPI
