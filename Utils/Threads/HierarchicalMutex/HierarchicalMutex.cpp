#include "HierarchicalMutex.hpp"

namespace RenderAPI
{
    namespace Thread
    {
        thread_local uint64 HierarchicalMutex::ThreadHierarchyValue(ULONG_MAX);

        HierarchicalMutex::~HierarchicalMutex()
        {
        }

        void HierarchicalMutex::CheckForHierarchyViolation()
        {
            if (ThreadHierarchyValue <= HierarchyValue)
            {
                throw std::logic_error("Mutex hierarchy violated!");
            }
        }
        void HierarchicalMutex::UpdateHierarchyValue()
        {
            PreviousHierarchyValue = ThreadHierarchyValue;
            ThreadHierarchyValue = HierarchyValue;
        }
        void HierarchicalMutex::Lock()
        {
            CheckForHierarchyViolation();
            InternalMutex.lock();
            UpdateHierarchyValue();
        }
        void HierarchicalMutex::Unlock()
        {
            if (ThreadHierarchyValue != HierarchyValue)
            {
                throw std::logic_error("Mutex hierarchy violated!");
            }
            ThreadHierarchyValue = PreviousHierarchyValue;
            InternalMutex.unlock();
        }
        bool HierarchicalMutex::TryLock()
        {
            CheckForHierarchyViolation();
            if (!InternalMutex.try_lock())
            {
                return false;
            }
            UpdateHierarchyValue();
            return true;
        }
    } // namespace Thread

} // namespace RenderAPI
