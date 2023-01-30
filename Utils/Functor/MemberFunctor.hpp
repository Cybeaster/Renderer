#pragma once
#include "../Types/SmartPtr.hpp"
#include "../Types/MemberFunctionType.hpp"

namespace RenderAPI
{
    struct TMemberFunctorBase
    {

        template <typename... ArgTypes>
        struct TTCallableInterface
        {
            virtual void Call(ArgTypes... Arguments) = 0;
            virtual ~TTCallableInterface() = default;
        };

        template <typename OwnerType, typename... ArgTypes>
        struct CallableBase : TTCallableInterface<ArgTypes...>
        {
            using FuncType = void (OwnerType::*)(ArgTypes...);
            CallableBase(OwnerType *OwnerPtr, FuncType Func) : Owner(MakeShared(OwnerPtr)),
                                                               Function(Func)
            {
            }

            virtual void Call(ArgTypes... Arguments) override
            {
                Owner->*Function(Arguments...);
            }

            TTSharedPtr<OwnerType> Owner;
            FuncType Function;
        };

        template <typename Owner, typename... ArgTypes>
        static TTSharedPtr<TTCallableInterface<ArgTypes...>> Create(Owner *Object, typename TTMemberFunctionType<Owner, void, ArgTypes...>::Type Function)
        {
            return MakeShared(new CallableBase<Owner, ArgTypes...>(Object, Function));
        }
    };
} // namespace RenderAPI