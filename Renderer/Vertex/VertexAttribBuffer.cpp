
#include "VertexAttribBuffer.hpp"
#include "../Renderer.hpp"

namespace RenderAPI
{

    TVertexAttribBuffer::TVertexAttribBuffer(const TVertexContext &Context) : VertexContext(Context)
    {
        RenderAPI::TRenderer::GetRenderer()->AddAttribBuffer(*this);
    }

    void TVertexAttribBuffer::EnableVertexAttribPointer()
    {
        VertexContext.BoundBuffer->Bind();
        glVertexAttribPointer(VertexContext.VertexIndex,
                              VertexContext.VertexSize,
                              VertexContext.VertexType,
                              VertexContext.IsNormalized,
                              VertexContext.VertexStride,
                              VertexContext.VertexPointer);

        glEnableVertexAttribArray(VertexContext.VertexAttribArrayIndex);
    }
} // namespace RenderAPI
