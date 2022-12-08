#pragma once
#include <Types/Types.hpp>
#include "Buffer.hpp"
namespace RenderAPI
{
    class TVertexArrayElem
    {

    public:
        TVertexArrayElem(TBuffer &&Buffer, uint32 Index, uint32 Size, bool Normalized, size_t Stride) noexcept
            : BoundBuffer(Move(Buffer)),
              VertexIndex(Index),
              VertexSize(Size),
              IsNormalized(Normalized),
              VertexStride(Stride)
        {
        }

        TVertexArrayElem() = delete;
        ~TVertexArrayElem() noexcept;

       // void DrawVertex(uint64 DrawType,uint32 FirstDrawIndex,  );
    private:
        TBuffer BoundBuffer;

        uint32 VertexIndex;
        uint32 VertexSize;
        bool IsNormalized;
        size_t VertexStride;
    };
} // namespace RenderAPI
