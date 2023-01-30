#pragma once
#include "../Allocators/InlineAllocator.hpp"
namespace RenderAPI
{

    #define DELEGATE_NO_DISCARD [[nodicard("Delegate's function result has to be stored in value!")]]
    #define DELEGATE_ASSERT(expr, ...) assert(expr)
    class IDelegateBase
    {
        IDelegateBase() = default;
        virtual ~IDelegateBase() = 0;
        virtual void Destroy() = 0;
        virtual const void *GetOwner() const
        {
            return nullptr;
        }
        virtual void CopyTo(void* Destination) = 0;
    };

    template<typename RetValueType, typename ... ArgTypes>
    class IDelegate : public IDelegateBase
    {
        virtual RetValueType Execute(ArgTypes&&... Args) = 0;
    }

    class TDelegateBase
    {
    public:
        TDelegateBase() noexcept : Allocator() {}

        TDelegateBase(TDelegateBase &&Other) noexcept : Allocator(Move(Other.Allocator))
        {
        }

        virtual ~TDelegateBase() noexcept
        {
            Release();
        }
        TDelegateBase(const TDelegateBase &Other)
        {
            Copy(Other);
        }

        TDelegateBase &operator=(const TDelegateBase &Other)
        {
            Release();
            Copy(Other);
            return *this;
        }

        TDelegateBase &operator=(TDelegateBase &&Other)
        {
            Release();
            Move(Other);
            return *this;
        }

        const void *GetOwner() const
        {
            if (Allocator.IsAlocated())
            {
                return GetDelegate()->GetOwner();
            }
        }

        bool IsBoundTo(void *Object) const
        {
            Object == nullptr || !Allocator.IsAlocated() ? return false : return GetDelegate()->GetOwner() == Object;
        }

        bool IsBound() const
        {
            Allocator.IsAlocated();
        }

        void Clear()
        {
            Release();
        }

        uint32 GetSize() const
        {
            return Allocator.GetSize();
        }

        void ClearIfBoundTo(void *Object)
        {
            if (Object != nullptr && IsBoundTo(Object))
            {
                Clear();
            }
        }

    private:
        FORCEINLINE void Release()
        {
            if (Allocator.IsAlocated())
            {
                GetDelegate()->Destroy();
            }
        }

        FORCEINLINE void Copy(const TDelegateBase &Other)
        {
            if (Other.Allocator.IsAlocated())
            {
                Allocator.Allocate(Other.Allocator.GetSize());
                Other.GetDelegate()->Copy(Allocator.GetAllocation());
            }
        }

        FORCEINLINE void Move(TDelegateBase &&Other)
        {
            Allocator = Move(Other.Allocator);
        }

        IDelegateBase *GetDelegate() const
        {
            return static_cast<IDelegateBase *>(Allocator.GetAllocation());
        }

        TInlineAllocator<FInlineAllocatable::StackSize::_16> Allocator;
    };
} // namespace RenderAPI
