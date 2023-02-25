#include "VertexArrayElem.hpp"

#include "Renderer/Renderer.hpp"

namespace RenderAPI
{

void OVertexArrayElem::DrawArrays() const
{
	// ORenderer::GetRenderer()->EnableBuffer(BoundBufferHandle);

	// GLCall(glEnable(DrawContext.Flag.Flag));
	GLCall(glFrontFace(DrawContext.FrontFace));

	GLCall(glDepthFunc(DrawContext.DepthFunction));

	GLCall(glDrawArrays(DrawContext.DrawType,
	                    DrawContext.FirstDrawIndex,
	                    DrawContext.DrawSize));
}

} // namespace RenderAPI
