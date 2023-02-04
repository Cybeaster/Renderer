#pragma once
#include "../Types/Types.hpp"
#include "Delegate.hpp"
namespace RenderAPI
{
#define INVALID_ID UINT32_MAX
    struct FDelegateHandle
    {

    public:
        constexpr FDelegateHandle() noexcept : LocalID(INVALID_ID) {}

        explicit FDelegateHandle(bool GenID) : LocalID(GenID ? GetNewID() : INVALID_ID)
        {
        }

        FDelegateHandle &operator=(const FDelegateHandle &Other) = default;
        FDelegateHandle::~FDelegateHandle() noexcept = default;

        FDelegateHandle(FDelegateHandle &&Other) : LocalID(Other.LocalID)
        {
            Other.Reset();
        }

        FDelegateHandle &operator=(FDelegateHandle &&Other) noexcept
        {
            LocalID = Other.LocalID;
            Other.Reset();
            return *this;
        }

        operator bool() const noexcept
        {
            return IsValid();
        }

        bool operator<(const FDelegateHandle &Other) const noexcept
        {
            return LocalID < Other.LocalID;
        }

        bool operator==(const FDelegateHandle &Other) const noexcept
        {
            return LocalID == Other.LocalID;
        }

        bool IsValid() const
        {
            return LocalID != INVALID_ID;
        }

        void Reset() noexcept
        {
            LocalID = INVALID_ID;
        }

    private:
        uint32 LocalID;
        static uint32 GlobalID;

        static uint32 GetNewID()
        {
            uint32 result = GlobalID++;
            if (GlobalID == INVALID_ID)
            {
                GlobalID = 0;
            }
            return result;
        }
    };

    template <typename... ArgTypes>
    class TMulticastDelegate
    {
    public:
        using DelegateType = TDelegate<void, ArgTypes...>;

    private:
        struct FDelegateHandlerPair
        {
            FDelegateHandle Handler;
            DelegateType Delegate;

            FORCEINLINE bool IsValid() const
            {
                return Handler.IsValid();
            }

            template <typename... Types>
            FORCEINLINE void Call(Types... Args)
            {
                Delegate.Execute(Forward<Args>(Args)...);
            }

            FORCEINLINE void Clear()
            {
                Delegate.Clear();
            }

            FORCEINLINE bool IsBoundTo(const FDelegateHandle& Other)
            {
                return Handler == Other;
            }

            FDelegateHandlerPair() : Handler(false)
            {
            }

            FDelegateHandlerPair(const FDelegateHandle &Handle, DelegateType &&Callback)
                : Handler(Handle),
                  Delegate(Move(Callback))
            {
            }
            FDelegateHandlerPair(const FDelegateHandle &Handle, const DelegateType &Callback)
                : Handler(Handle),
                  Delegate(Callback)
            {
            }
        };

        template <typename ObjectType, typename... PayloadArgs>
        using TConstMemberFunc =
            typename TTMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TConstFunction;

        template <typename ObjectType, typename... PayloadArgs>
        using TMemberFunc =
            typename TTMemberFunctionType<ObjectType, RetValueType, ArgTypes..., PayloadArgs...>::TFunction;

    public:
        constexpr TMulticastDelegate() : Locks(0) {}

        ~TMulticastDelegate() noexcept = default;

        TMulticastDelegate(const TMulticastDelegate &Other) = default;
        TMulticastDelegate(TMulticastDelegate &&Other) : Events(Move(Other.Events)),
                                                         Locks(Move(Other.Locks))
        {
        }

        TMulticastDelegate &operator=(const TMulticastDelegate &Delegate) = default;
        TMutlicastDelegate &operator=(TMulticastDelegate &&Delegate)
        {
            Events = Move(Delegate.Events);
            Locks = Move(Delegate.Locks);
        }

        template<typename ObjectType>
        bool RemoveFrom(ObjectType* Object)

        bool Remove(const FDelegateHandle& Handler)
        void IsBoundTo(FDelegateHandle& Handler);
        void RemoveAll();
        void Resize(const uint32 MaxSize = 0);
        void Broadcast(ArgTypes... Args);

        FORCEINLINE void GetSize() const
        {
            return Events.size();
        }

    private:
        void Lock()
        {
            ++Locks;
        }

        void Unlcok()
        {
            DELEGATE_ASSERT(Locks > 0);
            --Locks;
        }

        bool IsLocked()
        {
            return Locks > 0;
        }

        TVector<TDelegateHandlerPair> Events;
        uint32 Locks;
    };


    
        template<typename ObjectType>
        bool TMulticastDelegate::RemoveFrom(ObjectType* Object)
        {
            if(Object != nullptr)
            {
                for(auto event : Events)
                {
                    if(event.Delegate.GetOwner() == Object)
                    {
                        Events.
                    }
                }
            }
        }
        

} // namespace RenderAPI
