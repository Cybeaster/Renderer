#include "Delegate.hpp"

namespace RenderAPI
{
    template <typename TFunction>
    void TTDelegate::Bind(TFunction Function)
    {
        BoundFunctions.push_back(Function);
    }

    void TTDelegate::Execute(ArgTypes ... Args)
    {
        for(auto func : BoundFunctions)
        {
            func(Args...);
        }
    }
}