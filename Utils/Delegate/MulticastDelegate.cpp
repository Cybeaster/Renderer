#include "MulticastDelegate.hpp"

namespace RenderAPI
{

    void TMulticastDelegate::Resize(const uint32 MaxSize = 0)
    {
        if (!IsLocked())
        {
            uint32 toRemove = 0;
            for (uint32 it = 0; it < Events.size() - toRemove; it++)
            {
                if (!Events[it].IsValid())
                {
                    std::swap(Events[it], Events[toRemove]);
                    ++toRemove;
                }
            }
            if (toRemove > MaxSize)
            {
                Events.resize(Events.size() - toRemove);
            }
        }
    }

    void TMulticastDelegate::IsBoundTo(FDelegateHandle &Handler)
    {
        if(!IsLocked() && Handler.IsValid())
        {
            for(const auto& event : Events)
            {
                if(event.IsBoundTo(Handler))
                {
                    return true;
                }
            }
        }
        return false;
    }

    void TMulticastDelegate::RemoveAll()
    {
        if (!IsLocked())
        {
            for (auto handler : Events)
            {
                handler.Clear();
            }
        }
        else
        {
            Events.clear();
        }
    }

    void TMulticastDelegate::Broadcast(ArgTypes... Args)
    {
        Lock();
        for (auto event : Events)
        {
            if (event.IsValid())
            {
                event.Call(Args...);
            }
        }
        Unlock();
    }

} // namespace RenderAPI