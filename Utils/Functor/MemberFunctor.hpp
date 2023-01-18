#pragma once
#include "../Types/SmartPtr.hpp"
#include "../Types/MemberFunctionType.hpp"

namespace RenderAPI
{
    struct TMemberFunctorBase
    {

        template <typename... Args>
        struct TTCallableInterface
        {
            virtual void Call(Args... Arguments) = 0;
            virtual ~TTCallableInterface() = default;
        };

        template <typename OwnerType, typename... Args>
        struct CallableBase : TTCallableInterface<Args...>
        {
            using FuncType = void (OwnerType::*)(Args...);
            CallableBase(OwnerType *Owner, FuncType Func) : OwnerType(MakeShared(Owner)),
                                                            Function(Func)
            {
            }

            virtual void Call(Args... Arguments) override
            {
                Owner->*Function(Arguments...);
            }

            TTSharedPtr<OwnerType> Owner;
            FuncType Function;
        };

        template <typename Owner, typename... Args>
        static TTSharedPtr<TTCallableInterface<Args...>> Create(Owner *Object, typename TTMemberFunctionType<Owner, void, Args...>::Type Function)
        {
            return MakeShared(new CallableBase<Owner, Args...>(Object, Function));
        }
    };
} // namespace RenderAPI
