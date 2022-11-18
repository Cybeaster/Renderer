#pragma once
#include "../Types/Vector.hpp"
#include "../Types/BindableObject.hpp"
#include "../Types/SmartPtr.hpp"
#include "../Functor/Functor.hpp"
#include "../Types/MemberFunctionType.hpp"

namespace RenderAPI
{

    template <typename OwnerObject, typename RetType, typename... ArgTypes>
    class TTMemberDelegate
    {
        using FunctionType = ReturnType (OwnerObject::*)(ArgTypes...);

    public:
        TTMemberDelegate() = delete;

        template <typename ObjectType, typename RetType, typename... Args>
        static friend TTSharedPtr<TTMemberDelegate> Create(ObjectType *Object, MemberFunctionType<ObjectType, RetType, Args...>::Type FunctionType, Args... Arguments)
        {
            return MakeShared(new TTMemberDelegate(Object, FunctionType, Arguments));
        }

        void Bind(TBindableObjectInterface *Object, );
        void Unbind();
        void Execute()
        {
            for(auto& functor : BoundFunctors)
            {
                functor.Call();
            }
        }

    private:
        template <typename ObjectType, typename RetType, typename... Args, >
        TTMemberDelegate(ObjectType *Object, MemberFunctionType<ObjectType, RetType, Args...>::Type FunctionArg, Args... Arguments)
        {
            BoundFunctors.push_back({Object,FunctionArg,Arguments...});
        }

        TTVector<TTMemberFunctor<OwnerObject, RetType, ArgTypes...>> BoundFunctors;
    };

    template <typename ReturnType, typename... ArgTypes>
    class TTDelegate
    {
        using FunctionType = ReturnType(ArgTypes...);

    public:
        TTDelegate() = delete;

        template <typename RetType, typename... Args>
        static TTSharedPtr<TTDelegate> Create(RetType(FunctionType)(Args...), Args... Arguments)
        {
            return MakeShared(new TTDelegate(std::forward<RetType(Args...)>(FunctionType), std::forward<Args>(Arguments)...));
        }

        void Bind(TBindableObjectInterface *Object, );
        void Unbind();
        void Execute();

    private:
        template <typename RetType, typename... Args, >
        TTDelegate(FunctionType FunctionArg, Args... Arguments) : Function(FunctionArg, Arguments...) {}

        TTVector<TTWeakPtr<TBindableObjectInterface>> BoundObjects;
        TTFunctor<FunctionType> Function;
    };
} // namespace RenderAPI
