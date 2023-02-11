#pragma once

#include "DelegateBase.hpp"
#include "MemberFunctionType.hpp"
#include "SmartPtr.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"

namespace RenderAPI
{
template<typename IsConst, class Type, typename RetType, typename... Args2>
class OTRawDelegate;

template<typename IsConst, class ObjectType, typename RetType, typename... Args, typename... Args2>
class OTRawDelegate<IsConst, ObjectType, RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	using FunctionType = typename STMemberFunctionType<ObjectType, RetType, Args..., Args2...>::Type;

	OTRawDelegate(ObjectType* Object, FunctionType Function, Args2&&... Arguments)
	{
	}

	OTRawDelegate(ObjectType* Object, FunctionType Function, const OTuple<Args2...>& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Arguments)
	{
	}

	OTRawDelegate(ObjectType* Object, FunctionType Function, OTuple<Args2...>&& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Move(Arguments))
	{
	}

	RetType Execute(Args&&... NewPayload) override
	{
		return ExecuteImpl(Forward(NewPayload)..., TMakeIndexSequence<Args2...>());
	}

private:
	virtual RetType ExecuteImpl(Args2&&... NewPayload)
	{
	}

	OTWeakPtr<ObjectType> OwningObject;
	FunctionType* Callable;
	OTuple<Args2...> Payload;
};

} // namespace RenderAPI
