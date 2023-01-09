#include "VertexArrayElem.hpp"
#include "Renderer/Renderer.hpp"
namespace RenderAPI
{

    void TVertexArrayElem::DrawArrays() const
    {
        TRenderer::GetRenderer()->EnableBuffer(BoundBufferHandle);
        
        GLCall(glEnable(DrawContext.Flag));
        GLCall(glFrontFace(DrawContext.FrontFace));

        GLCall(glDepthFunc(DrawContext.DepthFunction));

        GLCall(glDrawArrays(DrawContext.DrawType,
                            DrawContext.FirstDrawIndex,
                            DrawContext.DrawSize));
    }

} // namespace RenderAPI
