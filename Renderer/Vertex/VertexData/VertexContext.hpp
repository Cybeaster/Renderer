#pragma once
#include "../Buffer.hpp"
#include "../SimpleVertexHandle.hpp"
#include "SmartPtr.hpp"
#include "Types.hpp"

namespace RenderAPI
{

struct SVertexContext
{
	SVertexContext(const SVertexContext& Context) = default;
	SVertexContext() = default;

	SVertexContext(SBufferHandle Handle,
	               const uint32 Index,
	               const uint32 Size,
	               const uint32 Type,
	               const bool Normalized,
	               const uint32 Stride,
	               const uint32 AttribArrayIndex,
	               void* Pointer)
	    : BoundBuffer(Move(Handle)), VertexIndex(Index), VertexSize(Size), VertexType(Type), IsNormalized(Normalized), VertexStride(Stride), VertexPointer(Pointer), VertexAttribArrayIndex(AttribArrayIndex)
	{
	}

	SVertexContext& operator=(const SVertexContext& Elem) = default;

	// vertex options
	SBufferHandle BoundBuffer;

	uint32 VertexIndex;
	uint32 VertexSize;
	uint32 VertexType;
	uint32 VertexStride;
	uint32 VertexAttribArrayIndex;
	void* VertexPointer;
	bool IsNormalized;
};
} // namespace RenderAPI
