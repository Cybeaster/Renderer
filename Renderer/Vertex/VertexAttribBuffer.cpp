
#include "VertexAttribBuffer.hpp"

#include "../Renderer.hpp"

namespace RenderAPI
{

TVertexAttribBuffer::OVertexAttribBuffer(const SVertexContext& Context)
    : VertexContext(Context)
{
	RenderAPI::ORenderer::GetRenderer()->AddAttribBuffer(*this);
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
