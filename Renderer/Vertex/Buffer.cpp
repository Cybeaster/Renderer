#include "Buffer.hpp"

#include "Renderer.hpp"

OBuffer::OBuffer(const void* Data, size_t Size)
{
	RAPI_LOG(Log, "Default Buffer initialization is used!");
	Init(Data, Size);
}

OBuffer::OBuffer(const SBufferContext& Context)
{
	BufferOption = Context.Option;
	BufferType = Context.Type;
	Init(Context.Data, Context.Size);
}

void OBuffer::Init(const void* Data, size_t Size)
{
	GLCall(glGenBuffers(1, &BufferID));
	GLCall(glBindBuffer(BufferType, BufferID));
	GLCall(glBufferData(BufferType, Size, Data, BufferOption));
}

OBuffer::~OBuffer()
{
	GLCall(glDeleteBuffers(1, &BufferID));
}

void OBuffer::Bind() const
{
	GLCall(glBindBuffer(BufferType, BufferID));
}
void OBuffer::Unbind() const
{
	GLCall(glBindBuffer(BufferType, 0));
}
