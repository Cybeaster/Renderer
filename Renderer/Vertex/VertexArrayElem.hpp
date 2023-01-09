#pragma once
#include <Types/Types.hpp>
#include <Hash.hpp>
#include "VertexData/DrawContext.hpp"
#include "Buffer.hpp"
#include "GL/glew.h"
#include <SmartPtr.hpp>
#include "SimpleVertexHandle.hpp"
namespace RenderAPI
{
    class TVertexArrayElem
    {
        using TBufferHandle = TBufferAttribVertexHandle;

    public:
        TVertexArrayElem(const TBufferHandle &Handle, const TDrawContext &Draw) noexcept
            : DrawContext(Draw),
              BoundBufferHandle(Handle)
        {
        }

        TVertexArrayElem(const TVertexArrayElem &Elem) noexcept
            : DrawContext(Elem.DrawContext),
              BoundBufferHandle(Elem.BoundBufferHandle)
        {
        }

        TVertexArrayElem() = default;
        ~TVertexArrayElem() noexcept
        {
        }

        TVertexArrayElem &operator=(const TVertexArrayElem &Elem)
        {
            DrawContext = Elem.DrawContext;
            BoundBufferHandle = Elem.BoundBufferHandle;
            return *this;
        }

        void DrawArrays() const;

    private:
        TBufferAttribVertexHandle BoundBufferHandle;
        TDrawContext DrawContext;
    };
} // namespace RenderAPI
