#pragma once
#include "Renderer/Vertex/Buffer.hpp"
#include "Renderer/Vertex/SimpleVertexHandle.hpp"
#include "Renderer/Vertex/VertexData/DrawContext.hpp"
#include "Utils/Types/HashMap/Hash.hpp"
#include "Utils/Types/SmartPtr.hpp"
#include "Utils/Types/Types.hpp"

namespace RenderAPI
{
class OVertexArrayElem
{
	using OBufferHandle = SBufferAttribVertexHandle;

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

	void Draw() const;

	SBufferAttribVertexHandle GetBoundBufferHandle() const
	{
		return BoundBufferHandle;
	}

private:
	SBufferAttribVertexHandle BoundBufferHandle;
	SDrawContext DrawContext;
};
} // namespace RenderAPI
