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
        TVertexAttribBuffer(/* args */);

        TVertexAttribBuffer(const TVertexContext &Context);

        ~TVertexAttribBuffer();

        void EnableVertexAttribPointer();

    private:
        TVertexContext VertexContext;
    };

    TVertexAttribBuffer::TVertexAttribBuffer(/* args */)
    {
    }

    TVertexAttribBuffer::~TVertexAttribBuffer()
    {
    }

} // namespace RendererAPI
