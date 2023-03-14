#pragma once

#include "DelegateBase.hpp"
#include "MemberFunctionType.hpp"
#include "SmartPtr.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
namespace RenderAPI
{
template<bool IsConst, typename ObjectType, typename RetType, typename... Args>
class OTSPDelegate
{
};

template<bool IsConst, typename ObjectType, typename RetType, typename... Args, typename... Args2>
class OTSPDelegate<IsConst, ObjectType, RetType(Args...), Args2...> : public OIDelegate<RetType, Args...>
{
public:
	using TSPDelegateFunction = typename STMemberFunctionType<IsConst, ObjectType, RetType, Args..., Args2...>::Type;

	OTSPDelegate(OWeakPtr<ObjectType> ObjectArg, TSPDelegateFunction FunctionArg, Args2&&... Arguments)
	    : Object(ObjectArg), Function(FunctionArg), Payload(Arguments...)
	{
	}

	OTSPDelegate(OSharedPtr<ObjectType> ObjectArg, TSPDelegateFunction FunctionArg, OTuple<Args2...> Arguments)
	    : Object(ObjectArg), Function(FunctionArg), Payload(Arguments)
	{
	}

	void CloneTo(void* Destination) override
	{
		new (Destination) OTSPDelegate(Object, Function, Payload.template Get<Args2...>());
	}

	RetType Execute(Args&&... Arguments)
	{
		return ExecuteImpl(Forward<Args>(Arguments)..., TMakeIndexSequence<Args2...>());
	}

private:
	template<size_t... Indices>
	RetType ExecuteImpl(Args&&... Arguments, TIndexSequenceWrapper<Indices...>)
	{
		if (Object.expired())
		{
			return RetType();
		}

		auto lockedObject = Object.lock();
		return (lockedObject.get()->*Function)(Forward(Arguments)..., (Payload.template Get<Indices>())...);
	}
	OWeakPtr<ObjectType> Object;
	TSPDelegateFunction Function;
	OTuple<Args2...> Payload;
};

} // namespace RenderAPI