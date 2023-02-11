#include "VertexArray.hpp"

#include "GL/glew.h"
#include "Renderer.hpp"
#include "glfw3.h"

namespace RenderAPI
{
TVertexArray::TVertexArray(/* args */)
{
}

void TVertexArray::AddVertexArray()
{
	TSimpleVertexIndex id;
	GLCall(glGenVertexArrays(1, &id.Index));
	GLCall(glBindVertexArray(id.Index));

	VertexIndicesArray.push_back(id);
}

TDrawVertexHandle TVertexArray::CreateVertexElement(const SVertexContext& VContext, const TDrawContext& RContext)
{
	++ElementsCounter;

	auto elementsHandle = TDrawVertexHandle(ElementsCounter);
	auto bufferAttribHandle = AddAttribBufferImpl(VContext);

	VertexElements[elementsHandle] = TVertexArrayElem(bufferAttribHandle, RContext);

	return elementsHandle;
}

TBufferAttribVertexHandle TVertexArray::AddAttribBufferImpl(const TVertexAttribBuffer& Buffer)
{
	++AttribBuffersCounter;
	auto bufferAttribHandle = TBufferAttribVertexHandle(AttribBuffersCounter);
	VertexAttribBuffers[bufferAttribHandle] = Buffer;

	return bufferAttribHandle;
}

TBufferAttribVertexHandle TVertexArray::AddAttribBuffer(const SVertexContext& VContext)
{
	return AddAttribBufferImpl(TVertexAttribBuffer(VContext));
}

TBufferAttribVertexHandle TVertexArray::AddAttribBuffer(const TVertexAttribBuffer& Buffer)
{
	return AddAttribBufferImpl(Buffer);
}

void TVertexArray::DrawArrays(const TDrawVertexHandle& Handle) const
{
	auto elem = VertexElements.find(Handle);
	elem->second.DrawArrays();
}

void TVertexArray::EnableBuffer(const TDrawVertexHandle& Handle)
{
	auto elem = VertexElements.find(Handle);
	EnableBuffer(elem->second.GetBoundBufferHandle());
}

void TVertexArray::EnableBuffer(const TBufferAttribVertexHandle& Handle)
{
	auto elem = VertexAttribBuffers[Handle];
	elem.EnableVertexAttribPointer();
}

TVertexArray::~TVertexArray()
{
}
} // namespace RenderAPI
