#pragma once
#include <Types.hpp>
class TBuffer
{

public:
    TBuffer(const void *Data, size_t size);

    TBuffer(const TBuffer& Buffer) : BufferID(Buffer.BufferID)
    {
    }

    TBuffer &operator=(const TBuffer &Buffer)
    {
        BufferID = Buffer.BufferID;
        return *this;
    }

    TBuffer() = default;

    ~TBuffer();

    void Bind() const;
    void Unbind() const;

private:
    uint32 BufferID;
};