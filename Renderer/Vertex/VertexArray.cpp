#include "VertexArray.hpp"

namespace RenderAPI
{
    TVertexArray::TVertexArray(/* args */)
    {
    }

    TVertexArrayHandle TVertexArray::CreateVertex(TBuffer &&Buffer, uint32 Index, uint32 Size, bool Normalized, size_t Stride)
    {
        ++VertexCounter;
        auto handle = TVertexArrayHandle(VertexCounter);
        VertexElements[handle] = TVertexArrayElem(Move(Buffer),Index,Size,Normalized,Stride);
        return handle;
    }

    TVertexArray::~TVertexArray()
    {
    }
} // namespace RenderAPI
