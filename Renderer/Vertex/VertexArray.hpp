#pragma once
#include "VertexArrayElem.hpp"
#include "Hash.hpp"
#include "Vector.hpp"
#include "VertexData/DrawContext.hpp"
#include "VertexData/VertexContext.hpp"

#include "SimpleVertexHandle.hpp"
#include "VertexAttribBuffer.hpp"
namespace RenderAPI
{
    struct TSimpleVertexIndex
    {
        explicit TSimpleVertexIndex(uint32 Value) : Index(Value)
        {
        }

        TSimpleVertexIndex() = default;
        uint32 Index = 0;
    };

    template <typename T>
    struct TTSimpleHandleHash
    {
        auto operator()(const T &FHandle) const
        {
            return GetHash(FHandle.GetHandle());
        }
    };

    class TVertexArray
    {
    public:
        TVertexArray(/* args */);
        ~TVertexArray();

        TDrawVertexHandle CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext);

        void DrawArrays(const TDrawVertexHandle &Handle) const;

        void EnableBuffer(const TBufferAttribVertexHandle &Handle);
        void EnableBuffer(const TDrawVertexHandle &Handle);

        void AddVertexArray();

        TBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer &Buffer);
        TBufferAttribVertexHandle AddAttribBuffer(const TVertexContext &VContext);

    private:
        TBufferAttribVertexHandle AddAttribBufferImpl(const TVertexAttribBuffer &Buffer);

        static inline uint64 ElementsCounter = 0;
        static inline uint64 AttribBuffersCounter = 0;

        TTHashMap<TDrawVertexHandle, TVertexArrayElem, TTSimpleHandleHash<TDrawVertexHandle>> VertexElements;
        TTHashMap<TBufferAttribVertexHandle, TVertexAttribBuffer, TTSimpleHandleHash<TBufferAttribVertexHandle>> VertexAttribBuffers;

        TTVector<TSimpleVertexIndex> VertexIndicesArray;
    };

};