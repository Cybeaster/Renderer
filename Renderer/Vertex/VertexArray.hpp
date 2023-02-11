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
struct STSimpleVertexIndex
{
	explicit STSimpleVertexIndex(uint32 Value)
	    : Index(Value)
	{
	}

	STSimpleVertexIndex() = default;
	uint32 Index = 0;
};

template<typename T>
struct STSimpleHandleHash
{
	auto operator()(const T& FHandle) const
	{
		return GetHash(FHandle.GetHandle());
	}
};

class OVertexArray
{
public:
	OVertexArray(/* args */);
	~OVertexArray();

	TDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext);

	void DrawArrays(const TDrawVertexHandle& Handle) const;

	void EnableBuffer(const OBufferAttribVertexHandle& Handle);
	void EnableBuffer(const TDrawVertexHandle& Handle);

	void AddVertexArray();

	OBufferAttribVertexHandle AddAttribBuffer(const TVertexAttribBuffer& Buffer);
	OBufferAttribVertexHandle AddAttribBuffer(const SVertexContext& VContext);

private:
	OBufferAttribVertexHandle AddAttribBufferImpl(const TVertexAttribBuffer& Buffer);

	static inline uint64 ElementsCounter = 0;
	static inline uint64 AttribBuffersCounter = 0;

	OTHashMap<TDrawVertexHandle, OVertexArrayElem, STSimpleHandleHash<TDrawVertexHandle>> VertexElements;
	OTHashMap<OBufferAttribVertexHandle, TVertexAttribBuffer, STSimpleHandleHash<OBufferAttribVertexHandle>> VertexAttribBuffers;

	OVector<STSimpleVertexIndex> VertexIndicesArray;
};

}; // namespace RenderAPI