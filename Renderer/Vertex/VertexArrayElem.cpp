#include "VertexArrayElem.hpp"
#include "Renderer/Renderer.hpp"
namespace RenderAPI
{

    void TVertexArrayElem::DrawArrays() const
    {
        //TRenderer::GetRenderer()->EnableBuffer(BoundBufferHandle);

        //GLCall(glEnable(DrawContext.Flag.Flag));
        //GLCall(glFrontFace(DrawContext.FrontFace.Flag));

      //  GLCall(glDepthFunc(DrawContext.DepthFunction.Flag));

        GLCall(glDrawArrays(DrawContext.DrawType,
                            DrawContext.FirstDrawIndex,
                            DrawContext.DrawSize));
    }

} // namespace RenderAPI
