#pragma once
#include "../Types/MemberFunctionType.hpp"
#include "DelegateBase.hpp"
#include "LambdaDelegate.hpp"
#include "RawDelegate.hpp"
#include "SPDelegate.hpp"
#include "SmartPtr.hpp"
#include "StaticDelegate.hpp"
#include "Types.hpp"
namespace RAPI
{

template<typename RetValueType, typename... ArgTypes>
class ODelegate : public ODelegateBase
{
private:
	template<typename ObjectType, typename... PayloadArgs>
	using TConstMemberFunc =
	    typename STMemberFunctionType<true, ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::Type;

	template<typename ObjectType, typename... PayloadArgs>
	using TNonConstMemberFunc =
	    typename STMemberFunctionType<false, ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::Type;

public:
	using TDelegateType = OIDelegate<RetValueType, ArgTypes...>;

#pragma region Static

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateRaw(ObjectType* Object, TNonConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
		delegate.Bind<OTRawDelegate<false, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(Object, Function, Forward<PayloadTypes>(Payload)...);
		return delegate;
	}

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateRaw(ObjectType* Object, TConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes... Payload)
	{
		ODelegate delegate;
		delegate.Bind<OTRawDelegate<true, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(Object, Function, Forward<PayloadTypes>(Payload)...);
		return delegate;
	}

#pragma endregion Static

	template<typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateStatic(RetValueType (*Function)(ArgTypes..., PayloadTypes...), PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTStaticDelegate<RetValueType, PayloadTypes...>>(Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

#pragma region SP

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateSP(OSharedPtr<ObjectType> ObjectArg, TNonConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTSPDelegate<false, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(ObjectArg, Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

	template<typename ObjectType, typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateSP(OSharedPtr<ObjectType> ObjectArg, TConstMemberFunc<ObjectType, PayloadTypes...> Function, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTSPDelegate<true, ObjectType, RetValueType(ArgTypes...), PayloadTypes...>>(ObjectArg, Function, Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

#pragma endregion SP

	template<typename LambdaType, typename... PayloadTypes>
	DELEGATE_NODISCARD static ODelegate CreateLambda(LambdaType&& Lambda, PayloadTypes&&... Args2)
	{
		ODelegate delegate;
		delegate.Bind<OTLambdaDelegate<LambdaType, RetValueType(ArgTypes...), PayloadTypes...>>(Forward(Lambda), Forward<PayloadTypes>(Args2)...);
		return delegate;
	}

	template<typename ObjectType, typename... Args2>
	void BindRaw(ObjectType* Object, TNonConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		*this = CreateRaw(Object, Function, Forward<Args2>(Args)...);
	}

	template<typename ObjectType, typename... Args2>
	void BindRaw(ObjectType* Object, TConstMemberFunc<ObjectType, Args2...> Function, Args2&&... Args)
	{
		*this = CreateRaw(Object, Function, Forward<Args2>(Args)...);
	}

	template<typename... Args2>
	void BindStatic(RetValueType(Function)(ArgTypes..., Args2...), Args2&&... Args)
	{
		*this = CreateStatic(Function, Forward<Args2>(Args)...);
	}

	template<typename LambdaType, typename... Payload>
	void BindLambda(LambdaType&& Lambda, Payload&&... Args2)
	{
		*this = CreateLambda(Forward<LambdaType>(Lambda), Forward<Payload>(Args2)...);
	}

	template<class ObjectType, typename... Payload>
	void BindSP(OSharedPtr<ObjectType> Object, TConstMemberFunc<ObjectType, Payload...> Function, Payload&&... Args2)
	{
		*this = CreateSP(Object, Function, Forward<Payload>(Args2)...);
	}

	template<class ObjectType, typename... Payload>
	void BindSP(OSharedPtr<ObjectType> Object, TNonConstMemberFunc<ObjectType, Payload...> Function, Payload&&... Args2)
	{
		*this = CreateSP(Object, Function, Forward<Payload>(Args2)...);
	}

	void Execute(ArgTypes&&... Args)
	{
		ASSERT(Allocator.IsAllocated());
		(static_cast<TDelegateType*>(GetDelegate()))->Execute(Forward<ArgTypes>(Args)...);
	}

private:
	template<typename ObjectType, typename... PayloadTypes>
	void Bind(PayloadTypes&&... Args)
	{
		Release();
		void* bunch = Allocator.Allocate(sizeof(ObjectType));
		new (bunch) ObjectType(Forward<PayloadTypes>(Args)...);
	}
};

} // namespace RAPI
