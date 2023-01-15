#pragma once
#include "Buffer.hpp"
#include <SmartPtr.hpp>
#include "VertexData/VertexContext.hpp"
#include <Types.hpp>
namespace RenderAPI
{
    class TVertexAttribBuffer
    {
    public:
        TVertexAttribBuffer(/* args */) = default;

        TVertexAttribBuffer(const TVertexContext &Context);

        void EnableVertexAttribPointer();

    private:
        TVertexContext VertexContext;
    };
} // namespace RendererAPI
