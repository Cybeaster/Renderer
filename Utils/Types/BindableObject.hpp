#pragma once

namespace RenderAPI
{
    class IBindableObject
    {
    public:
        virtual bool IsObjectValid() = 0;
    };

} // namespace RenderAPI
