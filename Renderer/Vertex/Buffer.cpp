#include "Buffer.hpp"
#include "Renderer.hpp"

TBuffer::TBuffer(const void *Data, size_t size)
{
    GLCall(glGenBuffers(1, &BufferID));
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, BufferID));
    GLCall(glBufferData(GL_ARRAY_BUFFER, size, Data, GL_STATIC_DRAW));
}

TBuffer::~TBuffer()
{
    GLCall(glDeleteBuffers(1, &BufferID));
}

void TBuffer::Bind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, BufferID));
}
void TBuffer::Unbind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}