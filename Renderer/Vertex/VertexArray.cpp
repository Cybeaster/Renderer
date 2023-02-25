#include "VertexArray.hpp"

#include "GL/glew.h"
#include "Renderer.hpp"
#include "glfw3.h"

namespace RenderAPI
{
OVertexArray::OVertexArray(/* args */)
{
}

void OVertexArray::AddVertexArray()
{
	STSimpleVertexIndex id;
	GLCall(glGenVertexArrays(1, &id.Index));
	GLCall(glBindVertexArray(id.Index));

	VertexIndicesArray.push_back(id);
}

TDrawVertexHandle OVertexArray::CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
{
	++ElementsCounter;

	auto elementsHandle = TDrawVertexHandle(ElementsCounter);
	auto bufferAttribHandle = AddAttribBufferImpl(VContext);

	VertexElements[elementsHandle] = OVertexArrayElem(bufferAttribHandle, RContext);

	return elementsHandle;
}

OBufferAttribVertexHandle OVertexArray::AddAttribBufferImpl(const OVertexAttribBuffer& Buffer)
{
	++AttribBuffersCounter;
	auto bufferAttribHandle = OBufferAttribVertexHandle(AttribBuffersCounter);
	VertexAttribBuffers[bufferAttribHandle] = Buffer;

	return bufferAttribHandle;
}

OBufferAttribVertexHandle OVertexArray::AddAttribBuffer(const SVertexContext& VContext)
{
	return AddAttribBufferImpl(OVertexAttribBuffer(VContext));
}

OBufferAttribVertexHandle OVertexArray::AddAttribBuffer(const OVertexAttribBuffer& Buffer)
{
	return AddAttribBufferImpl(Buffer);
}

void OVertexArray::DrawArrays(const TDrawVertexHandle& Handle) const
{
	auto elem = VertexElements.find(Handle);
	elem->second.DrawArrays();
}

void OVertexArray::EnableBuffer(const TDrawVertexHandle& Handle)
{
	auto elem = VertexElements.find(Handle);
	EnableBuffer(elem->second.GetBoundBufferHandle());
}

void OVertexArray::EnableBuffer(const OBufferAttribVertexHandle& Handle)
{
	auto elem = VertexAttribBuffers[Handle];
	elem.EnableVertexAttribPointer();
}

OVertexArray::~OVertexArray()
{
}
} // namespace RenderAPI
