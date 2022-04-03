#pragma once
#include <stdint.h>

class VertexBuffer;
class VertexBufferLayout;


class VertexArray
{
 public:
    VertexArray();
    ~VertexArray();

    void AddBuffer(const VertexBuffer& vb,const VertexBufferLayout& layout);
    void Bind() const;
    void Unbind() const;

 private:
   uint32_t m_RendererID;
};