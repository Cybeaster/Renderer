
#include "VertexAttribBuffer.hpp"
#include "glfw3.h"
#include "../Renderer.hpp"

namespace RenderAPI
{

    TVertexAttribBuffer::TVertexAttribBuffer(const TVertexContext &Context) : VertexContext(Context)
    {
        RenderAPI::TRenderer::GetRenderer()->AddAttribBuffer(*this);
    }

    void TVertexAttribBuffer::EnableVertexAttribPointer()
    {
        VertexContext.Buffer->Bind();
        glVertexAttribPointer(VertexContext.AttribIndex,
                              VertexContext.BunchSize,
                              VertexContext.Type,
                              VertexContext.IsNormalized,
                              VertexContext.Stride,
                              VertexContext.Pointer);

        glEnableVertexAttribArray(VertexContext.AttribArrayIndex);
    }
} // namespace RenderAPI
