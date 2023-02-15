#pragma once

#include "DelegateBase.hpp"
#include "MemberFunctionType.hpp"
#include "SmartPtr.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"

#include <vcruntime.h>

namespace RenderAPI
{
template<typename IsConst, class Type, typename RetType, typename... Args2>
class OTRawDelegate;

template<typename IsConst, class ObjectType, typename RetType, typename... Args, typename... Args2>
class OTRawDelegate<IsConst, ObjectType, RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	using TFunctionType = typename STMemberFunctionType<ObjectType, RetType, Args..., Args2...>::Type;

	OTRawDelegate(ObjectType* Object, TFunctionType Function, Args2&&... Arguments)
	{
	}

	OTRawDelegate(ObjectType* Object, FunctionType Function, const OTuple<Args2...>& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Arguments)
	{
	}

	OTRawDelegate(ObjectType* Object, TFunctionType Function, OTuple<Args2...>&& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Move(Arguments))
	{
	}

	RetType Execute(Args&&... NewPayload) override
	{
		return ExecuteImpl(Forward(NewPayload)..., TMakeIndexSequence<Args2...>());
	}

	void* GetOwner() override
	{
		return OwningObject.get();
	}

	void CopyTo(void* Destination)
	{
		new (Destination) OTRawDelegate(OwningObject, Callable, Payload);
	}

private:
	template<size_t... Sequence>
	RetType ExecuteImpl(Args2&&... NewPayload, TIndexSequence<Sequence...>)
	{
		OwningObject->*Callable(Forward(NewPayload)...,Payload.template Get<Sequence>()...);
	}

	OTWeakPtr<ObjectType> OwningObject;
	TFunctionType* Callable;
	OTuple<Args2...> Payload;
};

} // namespace RenderAPI
