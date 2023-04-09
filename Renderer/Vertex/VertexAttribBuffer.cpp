
#include "VertexAttribBuffer.hpp"

#include "../Renderer.hpp"

namespace RAPI
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
	RAPI::ORenderer::Get()->AddAttribBuffer(*this);
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
	RAPI::ORenderer::Get()->BindBuffer(VertexContext.BoundBuffer);
}

} // namespace RAPI
