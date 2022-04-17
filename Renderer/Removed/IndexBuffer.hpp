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
    {return m_Count;}

private:
    uint32_t m_RendererID;
    size_t m_Count;
};