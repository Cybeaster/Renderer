#pragma once
#include <Types.hpp>
class TBuffer
{

public:
    TBuffer(const void *Data, size_t size);

    TBuffer(TBuffer &&Buffer)
    {
        *this = Move(Buffer);
    }

    TBuffer &operator=(TBuffer &&Buffer)
    {
        BufferID = Buffer.BufferID;
        Buffer.BufferID = UINT32_MAX;
        return *this;
    }

    TBuffer() = default;
    ~TBuffer();

    void Bind() const;
    void Unbind() const;

private:
    uint32 BufferID;
};