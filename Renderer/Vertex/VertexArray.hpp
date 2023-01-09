#pragma once
#include "VertexArrayElem.hpp"
#include "Hash.hpp"
#include "Vector.hpp"
namespace RenderAPI
{

    struct TVertexIndex
    {
        uint32 Index = 0;
    };

    class TVertexArray
    {
    public:
        TVertexArray(/* args */);
        ~TVertexArray();

        TVertexArrayHandle CreateVertexElement(const TVertexContext &VContext, const TDrawContext &RContext);

        void DrawBuffer(const TVertexArrayHandle &Handle) const;
        void AddVertexArray();

    private:
        static inline uint64 VertexCounter = 0;

        TTHashMap<TVertexArrayHandle, TVertexArrayElem, TVertexArrayHandle::THandleHash> VertexElements;
        TTVector<TVertexIndex> VertexIndicesArray;
    };

};