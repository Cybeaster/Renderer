#pragma once
#include "../Types/Vector.hpp"
#include "../Types/BindableObject.hpp"
#include "../Types/SmartPtr.hpp"
#include "../Functor/Functor.hpp"
#include "../Types/MemberFunctionType.hpp"
#include "../Functor/MemberFunctor.hpp"

namespace RenderAPI
{

    template <typename... ArgTypes>
    class TTMemberDelegate
    {
        TTMemberDelegate() = default;

    public:
        template <typename ObjectType, typename... Args>
        void Bind(ObjectType *Object, typename TTMemberFunctionType<ObjectType, void, Args...>::Type Function)
        {
            BoundFunctors.push_back(TMemberFunctorBase::Create(Object, Function));
        }
        
        void Unbind();

        template <typename... FuncArgTypes>
        void Execute(FuncArgTypes... Args)
        {
            for (auto callable : BoundFunctors)
            {
                callable.Call(Args...);
            }
        }

    private:
        using SharedFunctor = TTSharedPtr<TMemberFunctorBase::TTCallableInterface<ArgTypes...>>;
        TTVector<SharedFunctor> BoundFunctors;
    };

    template <typename... ArgTypes>
    class TTDelegate
    {
        typedef void(FunctionType)(ArgTypes...);

    public:
        TTDelegate() = default;

        template <typename... Args>
        static TTSharedPtr<TTDelegate> Create(FunctionType Function, Args... Arguments)
        {
            return MakeShared(new TTDelegate(std::forward<RetType(Args...)>(FunctionType), std::forward<Args>(Arguments)...));
        }

        template <typename FunctionType>
        void Bind(FunctionType Function);

        template <class ObjectType, typename... Args>
        void Bind(ObjectType *Object, typename TTMemberFunctionType<ObjectType, void, Args...>::Type FunctionArg)
        {
            BoundFunctions.push_back(Function);
        }

        void Unbind();

        template <typename... FuncArgTypes>
        void Execute(FuncArgTypes... Args)
        {
            for (auto func : BoundFunctions)
            {
                func(Args...);
            }
        }

    private:
        TTVector<FunctionType *> BoundFunctions;
    };

} // namespace RenderAPI
