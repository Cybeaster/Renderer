#include "VertexArrayElem.hpp"

#include "Renderer/Renderer.hpp"

namespace RenderAPI
{

void OVertexArrayElem::Draw() const
{
	// ORenderer::GetRenderer()->EnableBuffer(BoundBufferHandle);

	if (DrawContext.Flag != UINT32_INVALID_VALUE)
	{
		GLCall(glEnable(DrawContext.Flag));
	}

	GLCall(glFrontFace(DrawContext.FrontFace));
	GLCall(glDepthFunc(DrawContext.DepthFunction));

	GLCall(glDrawArrays(DrawContext.DrawFlagType,
	                    DrawContext.FirstDrawIndex,
	                    DrawContext.DrawSize));
}

} // namespace RenderAPI
