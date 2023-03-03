#include "VertexArray.hpp"

#include "GL/glew.h"
#include "Renderer.hpp"
#include "glfw3.h"

namespace RenderAPI
{

void OVertexArray::AddVertexArray()
{
	STSimpleVertexIndex id;
	GLCall(glGenVertexArrays(1, &id.Index));
	GLCall(glBindVertexArray(id.Index));

	VertexIndicesArray.push_back(id);
}

SDrawVertexHandle OVertexArray::CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext)
{
	++ElementsCounter;

	auto elementsHandle = SDrawVertexHandle(ElementsCounter);
	auto bufferAttribHandle = AddAttribBufferImpl(OVertexAttribBuffer(VContext));

	VertexElements[elementsHandle] = OVertexArrayElem(bufferAttribHandle, RContext);

	return elementsHandle;
}

SBufferAttribVertexHandle OVertexArray::AddAttribBufferImpl(const OVertexAttribBuffer& Buffer)
{
	auto bufferAttribHandle = CreateNewVertexHandle();
	VertexAttribBuffers[bufferAttribHandle] = Buffer;
	return bufferAttribHandle;
}

SBufferAttribVertexHandle OVertexArray::AddAttribBufferImpl(OVertexAttribBuffer&& Buffer)
{
	auto bufferAttribHandle = CreateNewVertexHandle();
	VertexAttribBuffers[bufferAttribHandle] = Move(Buffer);
	return bufferAttribHandle;
}

SBufferAttribVertexHandle OVertexArray::AddAttribBuffer(const SVertexContext& VContext)
{
	return AddAttribBufferImpl(OVertexAttribBuffer(VContext));
}

SBufferAttribVertexHandle OVertexArray::AddAttribBuffer(const OVertexAttribBuffer& Buffer)
{
	return AddAttribBufferImpl(Buffer);
}

SBufferAttribVertexHandle OVertexArray::AddAttribBuffer(OVertexAttribBuffer&& Buffer)
{
	AddAttribBufferImpl(Move(Buffer));
}

SBufferAttribVertexHandle OVertexArray::AddAttribBuffer(SVertexContext&& VContext)
{
	return AddAttribBufferImpl(OVertexAttribBuffer(Move(VContext)));
}

void OVertexArray::Draw(const SDrawVertexHandle& Handle) const
{
	auto elem = VertexElements.find(Handle);
	elem->second.Draw();
}

void OVertexArray::EnableBufferAttribArray(const SDrawVertexHandle& Handle)
{
	auto elem = VertexElements.find(Handle);
	EnableBufferAttribArray(elem->second.GetBoundBufferHandle());
}

void OVertexArray::EnableBufferAttribArray(const SBufferAttribVertexHandle& Handle)
{
	auto elem = VertexAttribBuffers[Handle];
	elem.EnableVertexAttribPointer();
}

void OVertexArray::BindBuffer(const SBufferHandle& Handle)
{
	const auto& elem = BufferStorage[Handle];
	elem->Bind();
}

SBufferHandle OVertexArray::AddBuffer(const void* Data, size_t Size)
{
	auto bufferHandle = CreateNewBufferHandle();
	BufferStorage.insert({ bufferHandle, MakeShared<OBuffer>(Data, Size) });
	return bufferHandle;
}

SBufferHandle OVertexArray::AddBuffer(SBufferContext&& Context)
{
	auto bufferHandle = CreateNewBufferHandle();
	BufferStorage.insert({ bufferHandle, MakeShared<OBuffer>(Move(Context)) });
	return bufferHandle;
}

} // namespace RenderAPI
