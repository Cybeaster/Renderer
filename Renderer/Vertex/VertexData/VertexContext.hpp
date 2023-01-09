#pragma once
#include "Types.hpp"
#include "SmartPtr.hpp"
#include "../Buffer.hpp"
namespace RenderAPI
{

    struct TVertexContext
    {
        inline void Bind() const
        {
            BoundBuffer->Bind();
        }

        TVertexContext(const TVertexContext &Context) : BoundBuffer(Context.BoundBuffer),
                                                        VertexIndex(Context.VertexIndex),
                                                        VertexSize(Context.VertexSize),
                                                        VertexType(Context.VertexType),
                                                        IsNormalized(Context.IsNormalized),
                                                        VertexStride(Context.VertexStride),
                                                        VertexPointer(Context.VertexPointer),
                                                        VertexAttribArrayIndex(Context.VertexAttribArrayIndex)

        {
        }

        TVertexContext() = default;

        TVertexContext(TBuffer *Buffer,
                       const uint32 Index,
                       const uint32 Size,
                       const uint32 Type,
                       const bool Normalized,
                       const uint32 Stride,
                       const uint32 AttribArrayIndex,
                       void *Pointer) : BoundBuffer(Buffer),
                                        VertexIndex(Index),
                                        VertexSize(Size),
                                        VertexType(Type),
                                        IsNormalized(Normalized),
                                        VertexStride(Stride),
                                        VertexPointer(Pointer),
                                        VertexAttribArrayIndex(AttribArrayIndex)
        {
        }

        TVertexContext &operator=(const TVertexContext &Elem)
        {
            BoundBuffer = Elem.BoundBuffer;
            VertexIndex = Elem.VertexIndex;
            VertexSize = Elem.VertexSize;
            VertexType = Elem.VertexType;
            IsNormalized = Elem.IsNormalized;
            VertexStride = Elem.VertexStride;
            VertexPointer = Elem.VertexPointer;
            VertexAttribArrayIndex = Elem.VertexAttribArrayIndex;

            return *this;
        }

        // vertex options
        TTSharedPtr<TBuffer> BoundBuffer;
        uint32 VertexIndex;
        uint32 VertexSize;
        uint32 VertexType;
        uint32 VertexStride;
        uint32 VertexAttribArrayIndex;
        void *VertexPointer;
        bool IsNormalized;
    };
} // namespace RenderAPI
