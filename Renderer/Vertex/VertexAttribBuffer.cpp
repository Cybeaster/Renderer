
#include "VertexAttribBuffer.hpp"

#include "../Renderer.hpp"

namespace RenderAPI
{

OVertexAttribBuffer::OVertexAttribBuffer(const SVertexContext& Context)
    : VertexContext(Context)
{
	RegisterBuffer();
}

OVertexAttribBuffer::OVertexAttribBuffer(SVertexContext&& Context) noexcept
    : // NOLINT
    VertexContext(Move(Context))
{
	RegisterBuffer();
}

void OVertexAttribBuffer::RegisterBuffer()
{
	RenderAPI::ORenderer::Get()->AddAttribBuffer(*this);
}

void OVertexAttribBuffer::EnableVertexAttribPointer()
{
	BindBuffer();
	glVertexAttribPointer(VertexContext.VertexIndex,
	                      VertexContext.VertexSize,
	                      VertexContext.VertexType,
	                      VertexContext.IsNormalized,
	                      VertexContext.VertexStride,
	                      VertexContext.VertexPointer);

	glEnableVertexAttribArray(VertexContext.VertexAttribArrayIndex);
}
void OVertexAttribBuffer::BindBuffer()
{
	RenderAPI::ORenderer::Get()->BindBuffer(VertexContext.BoundBuffer);
}

} // namespace RenderAPI
