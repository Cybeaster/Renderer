#pragma once
#include "../Types/Vector.hpp"
#include "../Types/BindableObject.hpp"
#include "../Types/SmartPtr.hpp"
#include "../Functor/Functor.hpp"
#include "../Types/MemberFunctionType.hpp"

namespace RenderAPI
{

    // template <typename OwnerObject, typename... ArgTypes>
    // class TTMemberDelegate
    // {
    //     using FunctionType = void(OwnerObject::*)(ArgTypes...);

    // public:
    //     TTMemberDelegate() = delete;

    //     template <typename ObjectType, typename... Args>
    //     static TTSharedPtr<TTMemberDelegate> Create(ObjectType *Object, typename MemberFunctionType<ObjectType, void, Args...>::Type FunctionType, Args... Arguments)
    //     {
    //         return MakeShared(new TTMemberDelegate(Object, FunctionType, Arguments));
    //     }

    //     void Bind(IBindableObject *Object);
    //     void Unbind();
    //     void Execute()
    //     {
    //         for(auto& functor : BoundFunctors)
    //         {
    //             functor.Call();
    //         }
    //     }

    // private:
    //     template <typename ObjectType, typename... Args>
    //     TTMemberDelegate(ObjectType *Object, MemberFunctionType<ObjectType, void, Args...>::Type FunctionArg, Args... Arguments)
    //     {
    //         BoundFunctors.push_back({Object,FunctionArg,Arguments...});
    //     }

    //     TTVector<TTMemberFunctor<OwnerObject, void, ArgTypes...>> BoundFunctors;
    // };

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
        
        template <class ObjectType, typename Args>
        void Bind(ObjectType *Object, MemberFunctionType<ObjectType, void, Args...>::Type FunctionArg);

        void Unbind();
        
        void Execute(ArgTypes ... Args);

    private:
        TTVector<FunctionType> BoundFunctions;

    };
} // namespace RenderAPI
