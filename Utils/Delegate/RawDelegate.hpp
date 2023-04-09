#pragma once

#include "DelegateBase.hpp"
#include "MemberFunctionType.hpp"
#include "SmartPtr.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"

#include <vcruntime.h>

#pragma optimize("", off)

namespace RAPI
{
template<bool IsConst, class ObjectType, typename RetType, typename... Args2>
class OTRawDelegate;

template<bool IsConst, class ObjectType, typename RetType, typename... Args, typename... Args2>
class OTRawDelegate<IsConst, ObjectType, RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	using TFunctionType = typename STMemberFunctionType<IsConst, ObjectType, RetType, Args..., Args2...>::Type;

	OTRawDelegate(ObjectType* Object, TFunctionType Function, Args2&&... Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Forward<Args2>(Arguments)...)
	{
	}

	OTRawDelegate(ObjectType* Object, TFunctionType Function, const OTuple<Args2...>& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Arguments)
	{
	}

	OTRawDelegate(ObjectType* Object, TFunctionType Function, OTuple<Args2...>&& Arguments)
	    : OwningObject(Object), Callable(Function), Payload(Move(Arguments))
	{
	}

	RetType Execute(Args&&... NewPayload) override
	{
		return ExecuteImpl(Forward<Args>(NewPayload)..., TMakeIndexSequenceFor<Args2...>());
	}

	NODISCARD const void* GetOwner() const override
	{
		return OwningObject;
	}

	void CopyTo(void* Destination) override
	{
		new (Destination) OTRawDelegate(OwningObject, Callable, Payload);
	}

private:
	template<size_t... Sequence>
	RetType ExecuteImpl(Args&&... NewPayload, TIndexSequenceWrapper<Sequence...>)
	{
		(OwningObject->*Callable)(Forward<Args>(NewPayload)..., Payload.template Get<Sequence>()...);
	}

	ObjectType* OwningObject;
	TFunctionType Callable;
	OTuple<Args2...> Payload;
};

} // namespace RAPI
#pragma optimize("", on)