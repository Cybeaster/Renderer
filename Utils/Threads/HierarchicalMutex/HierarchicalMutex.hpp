#pragma once
#include <Types.hpp>
#include <Thread.hpp>
namespace RenderAPI
{
    namespace Thread
    {
        class HierarchicalMutex
        {

        public:
            explicit HierarchicalMutex(uint64 Value) : HierarchyValue(Value), PreviousHierarchyValue(0) {}

            void Lock();
            void Unlock();
            bool TryLock();

            HierarchicalMutex(/* args */) = delete;
            ~HierarchicalMutex() noexcept;

        private:
            void CheckForHierarchyViolation();
            void UpdateHierarchyValue();

            uint64 HierarchyValue;
            uint64 PreviousHierarchyValue;
            static thread_local uint64 ThreadHierarchyValue;
            TMutex InternalMutex;
        };
    } // namespace Thread

} // namespace RenderAPI