#pragma once
#include "../Buffer.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"

namespace RenderAPI
{

struct SVertexContext
{
	inline void Bind() const
	{
		BoundBuffer->Bind();
	}

	SVertexContext(const SVertexContext& Context) = default;
	SVertexContext() = default;

	SVertexContext(OBuffer* Buffer,
	               const uint32 Index,
	               const uint32 Size,
	               const uint32 Type,
	               const bool Normalized,
	               const uint32 Stride,
	               const uint32 AttribArrayIndex,
	               void* Pointer)
	    : BoundBuffer(Buffer), VertexIndex(Index), VertexSize(Size), VertexType(Type), IsNormalized(Normalized), VertexStride(Stride), VertexPointer(Pointer), VertexAttribArrayIndex(AttribArrayIndex)
	{
	}

	SVertexContext& operator=(const SVertexContext& Elem)
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
	OTSharedPtr<OBuffer> BoundBuffer;
	uint32 VertexIndex;
	uint32 VertexSize;
	uint32 VertexType;
	uint32 VertexStride;
	uint32 VertexAttribArrayIndex;
	void* VertexPointer;
	bool IsNormalized;
};
} // namespace RenderAPI
