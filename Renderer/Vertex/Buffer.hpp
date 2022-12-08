#pragma once
#include <Types.hpp>
class TBuffer
{

public:
    TBuffer(const void *Data, size_t size);

    TBuffer(TBuffer &&Buffer)
    {
        BufferID = Buffer.BufferID;
        Buffer.BufferID = UINT32_MAX;
    }
    ~TBuffer();

    void Bind() const;
    void Unbind() const;

private:
    uint32 BufferID;
};