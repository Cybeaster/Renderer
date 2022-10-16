#pragma once
#include <Assert.hpp>
#include <Tuple.hpp>
namespace RenderAPI
{
    namespace Functor
    {
        template <typename Func>
        class TTFunctor;

#pragma region FuncWithParams

        template <typename ReturnType, typename... Args>
        class TTFunctor<ReturnType(Args...)>
        {
            typedef ReturnType (*FuncType)(Args...);

        public:
            TTFunctor(FuncType &&FunctionType, TTuple<Args...>&& FuncArgs ) : Function(std::forward(FunctionArg)), Arguments(std::forward(FuncArgs)) {}

            TTFunctor &operator=(TTFunctor &&) = delete;
            TTFunctor &operator=(const TTFunctor &) = delete;
            TTFunctor(/* args */) = delete;

            ReturnType operator()(Args...)
            {
                CheckCallable();
                return (*Function)(Args...);
            }

            ~TTFunctor();

        private:
            void CheckCallable();

            FuncType Function;
            TTuple<Args...> Arguments;
        };

        template <typename ReturnType, typename... Args>
        inline TTFunctor<ReturnType(Args...)>::~TTFunctor()
        {
        }

        template <typename ReturnType, typename... Args>
        inline void TTFunctor<ReturnType(Args...)>::CheckCallable()
        {
            std::assert(Function);
        }
#pragma endregion FuncWithParams

#pragma region FuncWithoutParams
        template <typename ReturnType>
        class TTFunctor<ReturnType(void)>
        {
            typedef ReturnType (*FuncType)(void);

        public:
            TTFunctor(FuncType &&FunctionType) : Function(std::forward(FunctionType)) {}

            TTFunctor &operator=(TTFunctor &&) = delete;
            TTFunctor &operator=(const TTFunctor &) = delete;
            TTFunctor(/* args */) = delete;

            ReturnType operator()()
            {
                CheckCallable();
                return (*Function)();
            }

            ~TTFunctor();

        private:
            void CheckCallable();

            FuncType Function;
        };

        template <typename ReturnType>
        inline TTFunctor<ReturnType()>::~TTFunctor()
        {
        }

        template <typename ReturnType>
        inline void TTFunctor<ReturnType()>::CheckCallable()
        {
            ASSERT(Function);
        }

#pragma endregion FuncWithoutParams

    } // namespace Functor

} // namespace RenderAPI
