#pragma once
#include <Assert.hpp>
#include <Tuple.hpp>
namespace RenderAPI
{

    // template<typename FunctionType, typename OwnerObj, typename LastArgType>
    // auto Call(FunctionType OwnerObj::*MemberFunc, LastArgType LastArg) -> decltype()
    // {

    // }

    template <typename Func>
    class TTFunctor;

    template <typename ReturnType, typename... ArgTypes>
    class TTFunctor<ReturnType(ArgTypes...)>
    {
        typedef ReturnType(FuncType)(ArgTypes...);

    public:
        //template <typename FunConstructorType, typename... ArgConstructorTypes>
        TTFunctor(FunConstructorType &&FunctionType, ArgConstructorTypes... FuncArgs) : Function(std::forward<FunConstructorType>(FunctionType)), Arguments(std::forward<ArgConstructorTypes>(FuncArgs)...) {}

        TTFunctor(TTFunctor &&Functor)
        {
            Arguments = std::move(Functor.Arguments);
            Function = Functor.Function;
        }

        TTFunctor(const TTFunctor &Functor)
        {
            *this = Functor;
        }

        TTFunctor &operator=(const TTFunctor &Functor)
        {
            Arguments = Functor.Arguments;
            *Function = *Functor.Function;
            return *this;
        }

        TTFunctor &operator=(TTFunctor &&MovableFunctor)
        {
            Arguments = std::move(MovableFunctor.Arguments);
            Function = MovableFunctor.Function;
            MovableFunctor.Function = nullptr;
            MovableFunctor.Arguments.Empty();
            return this;
        }

        ReturnType operator()()
        {
            CheckCallable();
            return Arguments.Call<FuncType>(std::forward<FuncType>(*Function));
        }

        ~TTFunctor();

        template <typename FunctorType, typename... ParamTypes>
        static TTFunctor<ReturnType(ArgTypes...)> CreateFunctor(ParamTypes... Types)
        {
            return
        }

    private:
        void CheckCallable();

        TTuple<ArgTypes...> Arguments;
        FuncType *Function = nullptr;
    };

    template <typename ReturnType, typename... ArgTypes>
    inline TTFunctor<ReturnType(ArgTypes...)>::~TTFunctor()
    {
    }

    template <typename ReturnType, typename... ArgTypes>
    inline void TTFunctor<ReturnType(ArgTypes...)>::CheckCallable()
    {
        ASSERT(Function);
    }

    // #pragma region FuncWithOwner
    //         template <typename ReturnType, typename OwnerObject, typename... Args>
    //         class TTMemberFunctor : <ReturnType(OwnerObject::*)(Args...)>
    //         {
    //             using FuncType = ReturnType(OwnerObject::*)(Args...);
    //         public:
    //             TTMemberFunctor(FuncType &&FunctionType, Args &&...FuncArgs) : Function(std::forward<FuncType>(*FunctionType)), Arguments(FuncArgs...) {}
    //             ~TTMemberFunctor();

    //             void operator()()
    //             {

    //             }
    //         private:
    //             /* data */
    //         };
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