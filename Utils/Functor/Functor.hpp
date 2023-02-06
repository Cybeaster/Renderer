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
	using FuncType = ReturnType (ArgTypes...);

public:
	template<typename FuncConstrType, typename... ArgsConstrTypes>
	explicit OFunctor(FuncConstrType FunctionType, ArgsConstrTypes... FuncArgs)
	    : Function(FunctionType), Arguments(FuncArgs...)
	{
	}

	template<typename RetType, typename... LocalArgTypes>
	explicit OFunctor(OFunctor<RetType(LocalArgTypes...)>&& Functor)
	{
		Function = Functor.Function;
		Arguments = Move(Functor.Arguments);

		Functor.Arguments.Empty();
		Functor.Function = nullptr;
	}

	template<typename RetType, typename... Args>
	explicit OFunctor(const OFunctor<RetType(Args...)>& Functor)
	{
		Function = Functor.Function;
		Arguments = Functor.Arguments;
	}

	template<typename RetType, typename... Args>
	OFunctor& operator=(const OFunctor<RetType(Args...)>& Functor)
	{
		Function = Functor.Function;
		Arguments = Functor.Arguments;

		return *this;
	}

	template<typename RetType, typename... Args>
	OFunctor& operator=(OFunctor<RetType(Args...)>&& MovableFunctor)
	{
		Function = MovableFunctor.Function;
		Arguments = MovableFunctor.Arguments;

		MovableFunctor.Function = nullptr;
		MovableFunctor.Arguments.Empty();
		return *this;
	}

	ReturnType Call()
	{
		CheckCallable();
		return Arguments.template Call<FuncType>(std::forward<FuncType>(*Function));
	}

	ReturnType operator()()
	{
		return Call();
	}

	~OFunctor() = default;

protected:
	void CheckCallable()
	{
	}

	TTuple<ArgTypes...> Arguments;
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
//         TTSharedPtr<OwnerType> Owner;
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