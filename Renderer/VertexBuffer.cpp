#include "VertexBuffer.hpp"
#include "Renderer.hpp"


VertexBuffer::VertexBuffer(const void* Data, size_t size)
{
    GLCall(glGenBuffers(1,&m_RendererID));
    GLCall(glBindBuffer(GL_ARRAY_BUFFER,m_RendererID));
    GLCall(glBufferData(GL_ARRAY_BUFFER,size,Data,GL_STATIC_DRAW));
}

VertexBuffer::~VertexBuffer()
{
    GLCall(glDeleteBuffers(1,&m_RendererID));
}

void VertexBuffer::Bind()const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER,m_RendererID));

}
void VertexBuffer::Unbind()const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER,0));
}