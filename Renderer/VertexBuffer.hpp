#pragma once
#include <RenderAPI.hpp>
class VertexBuffer
{

public:
    VertexBuffer(const void *Data, size_t size);
    ~VertexBuffer();

    void Bind() const;
    void Unbind() const;

private:
    uint32 bufferID;
};