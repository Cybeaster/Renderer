#pragma once
#include "VertexArrayElem.hpp"
#include "Hash.hpp"
namespace RenderAPI
{

    class TVertexArray
    {
    public:
        TVertexArray(/* args */);
        ~TVertexArray();

        TVertexArrayHandle CreateVertex(TBuffer &&Buffer, uint32 Index, uint32 Size, bool Normalized, size_t Stride);
    private:
        static inline uint64 VertexCounter = 0;
        TTHashMap<TVertexArrayHandle,TVertexArrayElem> VertexElements;
    };

};