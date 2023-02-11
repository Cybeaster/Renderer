#pragma once
#include "Hash.hpp"
#include "SimpleVertexHandle.hpp"
#include "Vector.hpp"
#include "VertexArrayElem.hpp"
#include "VertexAttribBuffer.hpp"
#include "VertexData/DrawContext.hpp"
#include "VertexData/VertexContext.hpp"

namespace RenderAPI
{
struct TSimpleVertexIndex
{
	explicit TSimpleVertexIndex(uint32 Value)
	    : Index(Value)
	{
	}

	TSimpleVertexIndex() = default;
	uint32 Index = 0;
};

template<typename T>
struct TTSimpleHandleHash
{
	auto operator()(const T& FHandle) const
	{
		return GetHash(FHandle.GetHandle());
	}
};

class TVertexArray
{
public:
	TVertexArray(/* args */);
	~TVertexArray();

	TDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const TDrawContext& RContext);

	void DrawArrays(const TDrawVertexHandle& Handle) const;

	void EnableBuffer(const TBufferAttribVertexHandle& Handle);
	void EnableBuffer(const TDrawVertexHandle& Handle);

	void AddVertexArray();

	TBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer& Buffer);
	TBufferAttribVertexHandle AddAttribBuffer(const SVertexContext& VContext);

private:
	TBufferAttribVertexHandle AddAttribBufferImpl(const TVertexAttribBuffer& Buffer);

	static inline uint64 ElementsCounter = 0;
	static inline uint64 AttribBuffersCounter = 0;

	TTHashMap<TDrawVertexHandle, TVertexArrayElem, TTSimpleHandleHash<TDrawVertexHandle>> VertexElements;
	TTHashMap<TBufferAttribVertexHandle, TVertexAttribBuffer, TTSimpleHandleHash<TBufferAttribVertexHandle>> VertexAttribBuffers;

	OVector<TSimpleVertexIndex> VertexIndicesArray;
};

}; // namespace RenderAPI