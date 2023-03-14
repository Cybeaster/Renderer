#pragma once
#include "Hash.hpp"
#include "Renderer/Vertex/VertexArrayElements/VertexArrayElem.hpp"
#include "SimpleVertexHandle.hpp"
#include "Vector.hpp"
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
	OVertexArray(/* args */) = default;
	~OVertexArray() = default;

	SDrawVertexHandle CreateVertexElement(const SVertexContext& VContext, const SDrawContext& RContext);

	void Draw(const SDrawVertexHandle& Handle) const;

	void EnableBufferAttribArray(const SBufferAttribVertexHandle& Handle);
	/*
	 * Uses draw context to find attrib array.
	 * */
	void EnableBufferAttribArray(const SDrawVertexHandle& Handle);

	void AddVertexArray();

	SBufferAttribVertexHandle AddAttribBuffer(const OVertexAttribBuffer& Buffer);
	SBufferAttribVertexHandle AddAttribBuffer(const SVertexContext& VContext);

	SBufferAttribVertexHandle AddAttribBuffer(OVertexAttribBuffer&& Buffer);
	SBufferAttribVertexHandle AddAttribBuffer(SVertexContext&& VContext);

	SBufferHandle AddBuffer(const void* Data, size_t Size);
	SBufferHandle AddBuffer(SBufferContext&& Context);

	void BindBuffer(const SBufferHandle& Handle);

private:
	FORCEINLINE static SBufferAttribVertexHandle CreateNewVertexHandle()
	{
		++AttribBuffersCounter;
		return SBufferAttribVertexHandle(AttribBuffersCounter);
	}

	FORCEINLINE static SBufferHandle CreateNewBufferHandle()
	{
		++BufferCounter;
		return SBufferHandle(BufferCounter);
	}

	SBufferAttribVertexHandle AddAttribBufferImpl(const OVertexAttribBuffer& Buffer);
	SBufferAttribVertexHandle AddAttribBufferImpl(OVertexAttribBuffer&& Buffer);

	static inline uint64 ElementsCounter = 0;
	static inline uint64 AttribBuffersCounter = 0;
	static inline uint64 BufferCounter = 0;

	OTHashMap<SDrawVertexHandle, OVertexArrayElem, STSimpleHandleHash<SDrawVertexHandle>> VertexElements;
	OTHashMap<SBufferAttribVertexHandle, OVertexAttribBuffer, STSimpleHandleHash<SBufferAttribVertexHandle>> VertexAttribBuffers;
	OTHashMap<SBufferHandle, OSharedPtr<OBuffer>, STSimpleHandleHash<SBufferHandle>> BufferStorage;

	OTVector<STSimpleVertexIndex> VertexIndicesArray;
};

}; // namespace RenderAPI