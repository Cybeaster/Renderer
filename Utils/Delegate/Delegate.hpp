#pragma once
#include "../Types/MemberFunctionType.hpp"
#include "DelegateBase.hpp"
#include "LambdaDelegate.hpp"
#include "LamdaDelegate.hpp"
#include "RawDelegate.hpp"
#include "SPDelegate.hpp"
#include "StaticDelegate.hpp"
#include "Types.hpp"
namespace RenderAPI
{

template<typename RetValueType, typename... ArgTypes>
class ODelegate : public ODelegateBase
{
private:
	template<typename ObjectType, typename... PayloadArgs>
	using TConstMemberFunc =
	    typename STMemberFunctionType<true, ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TConstFunction;

	template<typename ObjectType, typename... PayloadArgs>
	using TNonConstMemberFunc =
	    typename STMemberFunctionType<false, ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TFunction;

public:
	using TDelegateType = OIDelegate<RetValueType, ArgTypes...>;

#pragma region Static

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateRaw(ObjectType* Object, TNonConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
		delegate.Bind<OTRawDelegate<false, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(Object, Function, Forward<PayloadTypes>(Payload)...);
		return delegate;
	}

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateRaw(ObjectType* Object, TConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
		delegate.Bind<OTRawDelegate<true, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(Object, Function, Forward<PayloadTypes>(Payload)...);
		return delegate;
	}

#pragma endregion Static

	template<typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateStatic(RetValueType (*Function)(ArgTypes..., PayloadTypes...), PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTStaticDelegate<RetValueType, PayloadTypes...>>(Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

#pragma region SP

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateSP(OTSharedPtr<ObjectType> ObjectArg, TNonConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTSPDelegate<false, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(ObjectArg, Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateSP(OTSharedPtr<ObjectType> ObjectArg, TConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTSPDelegate<true, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(ObjectArg, Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

#pragma endregion SP

	template<typename LambdaType, typename... PayloadTypes>
	DELEGATE_NO_DISCARD static ODelegate CreateLambda(LambdaType&& Lambda, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTLambdaDelegate<LambdaType, RetValueType(ArgTypes...), PayloadTypes...>>(Forward(Lambda), Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

	void Execute(ArgTypes&&... Args)
	{
		DELEGATE_ASSERT(Allocator.IsAllocated());
		(static_cast<TDelegateType*>(GetDelegate()))->Execute(Forward<ArgTypes>(Args)...);
	}

private:
	template<typename ObjectType, typename... PayloadTypes>
	void Bind(PayloadTypes&&... Args)
	{
		Release();
		void* bunch = Allocator.Allocate(sizeof(ObjectType));
		new (bunch) ObjectType(std::forward(Args...));
	}
};

} // namespace RenderAPI
