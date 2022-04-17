#pragma once
#include <Renderer.hpp>
#include <vector>
#include "GL/glew.h"
struct VertexBufferElement
{
    uint32_t type;
    uint32_t count;
    unsigned char normalized;

    static uint32_t GetSizeOfType(uint32_t type)
    {
        switch (type)
        {
            case GL_FLOAT: return 4;
            case GL_UNSIGNED_INT: return 4;
            case GL_UNSIGNED_BYTE: return 1;
        }
        ASSERT(false);
        return 0;
    }
};


class VertexBufferLayout
{

public:
    VertexBufferLayout();
    ~VertexBufferLayout();

    template <typename T>
    void Push(uint32_t count)
    {
        static_assert(false)
    }
    template<>
    void Push<float>(uint32_t count)
    {
        m_Elements.push_back({GL_FLOAT,count,GL_FALSE});
        m_Stride += VertexBufferElement::GetSizeOfType(GL_FLOAT) * count;
    }

    template<>
    void Push<uint32_t>(uint32_t count)
    {
        m_Elements.push_back({GL_UNSIGNED_INT,count,GL_FALSE});
        m_Stride += VertexBufferElement::GetSizeOfType(GL_UNSIGNED_INT) * count ; 
    }
    template<>
    void Push<uint8_t>(uint32_t count)
    {
        m_Elements.push_back({GL_UNSIGNED_BYTE,count,GL_TRUE});
        m_Stride += VertexBufferElement::GetSizeOfType(GL_UNSIGNED_BYTE) * count; 
    }

    inline uint32_t GetStride() const
    {return m_Stride;}

    inline const auto& GetElements() const
    {return m_Elements;}
private:
    std::vector<VertexBufferElement> m_Elements;
    uint32_t m_Stride;
    
  

};


