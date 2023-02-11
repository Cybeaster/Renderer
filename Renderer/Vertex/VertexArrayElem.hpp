#pragma once
#include "Buffer.hpp"
#include "SimpleVertexHandle.hpp"
#include "VertexData/DrawContext.hpp"

#include <Hash.hpp>
#include <SmartPtr.hpp>
#include <Types/Types.hpp>

namespace RenderAPI
{
class OVertexArrayElem
{
	using OBufferHandle = OBufferAttribVertexHandle;

public:
	OVertexArrayElem(const OBufferHandle& Handle, const SDrawContext& Draw) noexcept
	    : DrawContext(Draw), BoundBufferHandle(Handle)
	{
	}

	OVertexArrayElem(const OVertexArrayElem& Elem) noexcept
	    : DrawContext(Elem.DrawContext), BoundBufferHandle(Elem.BoundBufferHandle)
	{
	}

	OVertexArrayElem() = default;
	~OVertexArrayElem() noexcept
	{
	}

	OVertexArrayElem& operator=(const OVertexArrayElem& Elem)
	{
		DrawContext = Elem.DrawContext;
		BoundBufferHandle = Elem.BoundBufferHandle;
		return *this;
	}

	void DrawArrays() const;

	OBufferAttribVertexHandle GetBoundBufferHandle() const
	{
		return BoundBufferHandle;
	}

private:
	OBufferAttribVertexHandle BoundBufferHandle;
	SDrawContext DrawContext;
};
} // namespace RenderAPI
