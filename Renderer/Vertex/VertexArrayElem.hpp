#pragma once
#include <Types/Types.hpp>
#include "Buffer.hpp"
namespace RenderAPI
{
    class TVertexArrayHandle
    {
    public:
        TVertexArrayHandle(uint64 ID) : Handle(ID)
        {
        }

        uint64 GetHandle() const
        {
            return Handle;
        }

        bool operator==(const TVertexArrayHandle &ArrayHandle)
        {
            return Handle == ArrayHandle.Handle;
        }

    private:
        uint64 Handle;
    };

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

        TVertexArrayElem(TVertexArrayElem &&Elem)
        {
            *this = Move(Elem);
        }

        TVertexArrayElem &operator=(TVertexArrayElem &&Elem)
        {
            BoundBuffer = Move(Elem.BoundBuffer);
            BoundBuffer = Move(Elem.BoundBuffer);
            VertexIndex = Elem.VertexIndex;
            VertexSize = Elem.VertexSize;
            IsNormalized = Elem.IsNormalized;
            VertexStride = Elem.VertexStride;
        }

        // void DrawVertex(uint64 DrawType,uint32 FirstDrawIndex,  );
    private:
        TBuffer BoundBuffer;

        uint32 VertexIndex;
        uint32 VertexSize;
        bool IsNormalized;
        size_t VertexStride;
    };
} // namespace RenderAPI
