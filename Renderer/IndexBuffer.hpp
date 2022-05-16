#pragma once
#include "cstdint"

class IndexBuffer
{

public:
    IndexBuffer(const uint32_t* data, size_t count);
    ~IndexBuffer();


    void Bind() const;
    void Unbind() const;

    inline uint32_t GetCount() const
    {return count;}

private:
    uint32_t bufferID;
    size_t count;
};