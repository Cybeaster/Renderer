#include "Buffer.hpp"

#include "Renderer.hpp"


OBuffer::OBuffer(const void* Data, size_t size)
{
	GLCall(glGenBuffers(1, &BufferID));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, BufferID));
	GLCall(glBufferData(GL_ARRAY_BUFFER, size, Data, GL_STATIC_DRAW));
}

OBuffer::~OBuffer()
{
	GLCall(glDeleteBuffers(1, &BufferID));
}

void OBuffer::Bind() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, BufferID));
}
void OBuffer::Unbind() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}