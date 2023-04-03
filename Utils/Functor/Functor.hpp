#pragma once
#include <Assert.hpp>
#include <SmartPtr.hpp>
#include <Tuple.hpp>
#include <stdexcept>
#include <typeinfo>

namespace RenderAPI
{

template<typename Func>
class OFunctor;

struct SFunctorBase
{
	struct SICallableInterface
	{
		virtual void Call() = 0;
		virtual ~SICallableInterface() = default;
	};

	template<typename CallableType>
	struct SCallableImpl : public SICallableInterface
	{
		explicit SCallableImpl(CallableType&& FunctorTypeArg) noexcept
		    : FunctorArg(Move(FunctorTypeArg))
		{
		}

		void Call() override
		{
			FunctorArg.Call();
		}
		CallableType FunctorArg;
	};

	template<typename RetType, typename... ArgTypes>
	inline static SICallableInterface* Create(RetType(Function)(ArgTypes...), ArgTypes... FuncArgs)
	{
		using FunctorType = OFunctor<RetType(ArgTypes...)>;
		return new SCallableImpl<FunctorType>(FunctorType(Function, FuncArgs...));
	}
};

template<typename ReturnType, typename... ArgTypes>
class OFunctor<ReturnType(ArgTypes...)> : public SFunctorBase
{
	struct TCallableInterface;
	using FuncType = ReturnType(ArgTypes...);

public:
	OFunctor() = default;
	~OFunctor() = default;

	OFunctor& operator=(const OFunctor& Functor) = default;
	OFunctor(const OFunctor<RetType(ArgTypes...)> Functor) = default;

	template<typename FuncConstrType, typename... ArgsConstrTypes>
	explicit OFunctor(FuncConstrType FunctionType, ArgsConstrTypes... FuncArgs)
	    : Function(FunctionType)
	    , Payload(FuncArgs...)
	{
	}

	template<typename RetType, typename... LocalArgTypes>
	explicit OFunctor(OFunctor<RetType(LocalArgTypes...)>&& Functor)
	{
		Function = Functor.Function;
		Payload = Move(Functor.Payload);

		Functor.Payload.Empty();
		Functor.Function = nullptr;
	}

	template<typename RetType, typename... Args>
	OFunctor& operator=(OFunctor<RetType(Args...)>&& MovableFunctor)
	{
		Function = MovableFunctor.Function;
		Payload = MovableFunctor.Payload;

		MovableFunctor.Function = nullptr;
		MovableFunctor.Payload.Empty();
		return *this;
	}

	template<typename... Args>
	ReturnType Call(Args&&... Arguments)
	{
		CheckCallable();
		return Payload.template Call<FuncType>(std::forward<FuncType>(*Function), Forward(Arguments)...);
	}

	template<typename... Args>
	ReturnType operator()(Args&&... Arguments)
	{
		return Call(Forward(Arguments)...);
	}

protected:
	void CheckCallable()
	{
	}

	OTuple<ArgTypes...> Payload;
	FuncType* Function;
};

// #pragma region FuncWithOwner
//     template <typename OwnerType, typename ReturnType, typename... Args>
//     class TTMemberFunctor : public TTFunctor<ReturnType (OwnerType::*)(Args...)>
//     {
//         using FuncType = ReturnType (OwnerType::*)(Args...);

//     public:
//         template <typename OwnerType, typename FunConstructorType, typename... ArgConstructorTypes>
//         TTMemberFunctor(OwnerType *OwnerArg, FunConstructorType &&FunctionType, ArgConstructorTypes... FuncArgs)
//             : Owner(MakeShared<OwnerType>(OwnerArg)),
//               Function(std::forward<FunConstructorType>(FunctionType)),
//               Arguments(std::forward<ArgConstructorTypes>(FuncArgs)...) {}

//         ~TTMemberFunctor();

//         void operator()()
//         {
//             CheckCallable();
//             Arguments.Call(std::forward<FuncType>(*Function), *Owner);
//         }

//     private:
//         OSharedPtr<OwnerType> Owner;
//     };
// #pragma endregion FuncWithOwner

} // namespace RenderAPI

#define DECLARE_FUNCTOR(FunctorTypeName, RetValue) \
	using(FunctorTypeName) = RenderAPI::TTFunctor<RetValue(void)>;

#define DECLARE_FUNCTOR_OneParam(FunctorTypeName, RetValue, Arg) \
	using(FunctorTypeName) = RenderAPI::TTFunctor<RetValue(Arg)>;

#define DECLARE_FUNCTOR_TwoParams(FunctorTypeName, RetValue, Arg1, Arg2) \
	using(FunctorTypeName) = RenderAPI::TTFunctor<RetValue(Arg1, Arg2)>;

#define DECLARE_FUNCTOR_ThreeParams(FunctorTypeName, RetValue, Arg1, Arg2, Arg3) \
	using(FunctorTypeName) = RenderAPI::TTFunctor<RetValue(Arg1, Arg2, Arg3)>;   \
	\