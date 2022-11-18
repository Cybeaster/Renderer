#pragma once
#include <Assert.hpp>
#include <Tuple.hpp>
#include <stdexcept>
#include <typeinfo>
#include <SmartPtr.hpp>
namespace RenderAPI
{

    template <typename Func>
    class TTFunctor;

    struct TFunctorBase
    {
        struct TCallableInterface
        {
            virtual void Call() = 0;
            virtual ~TCallableInterface() = default;
        };

        template <typename CallableType>
        struct TTCallableImpl : public TCallableInterface
        {
            template <typename FunctorType>
            TTCallableImpl(FunctorType&& FunctorTypeArg) : FunctorArg(FunctorTypeArg)
            {
            }

            void Call() override
            {
                FunctorArg.Call();
            }
            CallableType FunctorArg;
        };

        template <typename RetType, typename... ArgTypes>
        inline static TCallableInterface *Create(RetType(Function)(ArgTypes...), ArgTypes... FuncArgs)
        {
            using FunctorType = TTFunctor<RetType(ArgTypes...)>;
            return new TTCallableImpl<FunctorType>(FunctorType(Function, FuncArgs...));
        }
    };

    template <typename ReturnType, typename... ArgTypes>
    class TTFunctor<ReturnType(ArgTypes...)> : public TFunctorBase
    {
        struct TCallableInterface;
        typedef ReturnType(FuncType)(ArgTypes...);

    public:
        template <typename FuncConstrType, typename... ArgsConstrTypes>
        TTFunctor(FuncConstrType FunctionType, ArgsConstrTypes... FuncArgs) : Function(FunctionType), Arguments(FuncArgs...) {}

        template <typename RetType, typename... Args>
        TTFunctor(TTFunctor<RetType(Args...)> &&Functor)
        {
            Function = Functor.Function;
            Arguments = Move(Functor.Arguments);

            Functor.Arguments.Empty();
            Functor.Function = nullptr;
        }

        template <typename RetType, typename... Args>
        TTFunctor(const TTFunctor<RetType(Args...)> &Functor)
        {
            Function = Functor.Function;
            Arguments = Functor.Arguments;
        }

        template <typename RetType, typename... Args>
        TTFunctor &operator=(const TTFunctor<RetType(Args...)> &Functor)
        {
            Function = Functor.Function;
            Arguments = Functor.Arguments;

            return *this;
        }

        template <typename RetType, typename... Args>
        TTFunctor &operator=(TTFunctor<RetType(Args...)> &&MovableFunctor)
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
            return Arguments.Call<FuncType>(std::forward<FuncType>(*Function));
        }

        ReturnType operator()()
        {
            return Call();
        }

        ~TTFunctor() = default;

    protected:
        void CheckCallable()
        {
        }

        TTuple<ArgTypes...> Arguments;
        FuncType *Function;
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
    using FunctorTypeName = RenderAPI::TTFunctor<RetValue(void)>;

#define DECLARE_FUNCTOR_OneParam(FunctorTypeName, RetValue, Arg) \
    using FunctorTypeName = RenderAPI::TTFunctor<RetValue(Arg)>;

#define DECLARE_FUNCTOR_TwoParams(FunctorTypeName, RetValue, Arg1, Arg2) \
    using FunctorTypeName = RenderAPI::TTFunctor<RetValue(Arg1, Arg2)>;

#define DECLARE_FUNCTOR_ThreeParams(FunctorTypeName, RetValue, Arg1, Arg2, Arg3) \
    using FunctorTypeName = RenderAPI::TTFunctor<RetValue(Arg1, Arg2, Arg3)>;    \
    \