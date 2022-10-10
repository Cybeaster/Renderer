#pragma once
#include <RenderAPITypes.hpp>
class IndexBuffer
{

public:
    IndexBuffer(const uint32 *data, size_t count);
    ~IndexBuffer();

    void Bind() const;
    void Unbind() const;

    inline uint32 GetCount() const
    {
        return count;
    }

private:
    uint32 bufferID;
    size_t count;
};