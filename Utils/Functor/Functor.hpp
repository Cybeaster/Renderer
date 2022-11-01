#pragma once
#include <Assert.hpp>
#include <Tuple.hpp>
namespace RenderAPI
{
    namespace Functor
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
            TTFunctor() = default;

            TTFunctor(TTFunctor &&Functor)
            {
                Arguments = std::move(MovableFunctor.Arguments);
                Function = MovableFunctor.Function;
            }

            TTFunctor(FuncType &&FunctionType, ArgTypes... FuncArgs) : Function(std::forward<FuncType>(FunctionType)), Arguments(std::forward<ArgTypes>(FuncArgs)...) {}

            TTFunctor &operator=(const TTFunctor &)
            {
                return this;
            }

            TTFunctor &operator=(TTFunctor &&MovableFunctor)
            {
                Arguments = std::move(MovableFunctor.Arguments);
                Function = MovableFunctor.Function;
                return this;
            }

            ReturnType operator()()
            {
                CheckCallable();
                return Arguments.Call<FuncType>(std::forward<FuncType>(*Function));
            }

            ~TTFunctor();

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

    } // namespace Functor

} // namespace RenderAPI

#define DECLARE_FUNCTOR(FunctorTypeName, RetValue) \
    using FunctorTypeName = RenderAPI::Functor::TTFunctor<RetValue(void)>;

#define DECLARE_FUNCTOR_OneParam(FunctorTypeName, RetValue, Arg) \
    using FunctorTypeName = RenderAPI::Functor::TTFunctor<RetValue(Arg)>;

#define DECLARE_FUNCTOR_TwoParams(FunctorTypeName, RetValue, Arg1, Arg2) \
    using FunctorTypeName = RenderAPI::Functor::TTFunctor<RetValue(Arg1, Arg2)>;

#define DECLARE_FUNCTOR_ThreeParams(FunctorTypeName, RetValue, Arg1, Arg2, Arg3)       \
    using FunctorTypeName = RenderAPI::Functor::TTFunctor<RetValue(Arg1, Arg2, Arg3)>; \
    \