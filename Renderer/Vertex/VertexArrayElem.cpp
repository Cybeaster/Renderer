#include "VertexArrayElem.hpp"
#include "Renderer/Renderer.hpp"
namespace RenderAPI
{

    void TVertexArrayElem::DrawBuffer() const
    {
        VertexContext.Bind();

        GLCall(glVertexAttribPointer(
            VertexContext.VertexIndex,
            VertexContext.VertexSize,
            VertexContext.VertexType,
            VertexContext.IsNormalized,
            VertexContext.VertexStride,
            0));

        GLCall(glEnableVertexAttribArray(DrawContext.AttributeArray));
        GLCall(glEnable(DrawContext.Flag));
        GLCall(glFrontFace(DrawContext.FrontFace));

        GLCall(glDepthFunc(DrawContext.DepthFunction));

        GLCall(glDrawArrays(DrawContext.DrawType,
                            DrawContext.FirstDrawIndex,
                            DrawContext.DrawSize));
    }

} // namespace RenderAPI
