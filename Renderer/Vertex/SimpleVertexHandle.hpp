#pragma once
#include <Types.hpp>

struct SVertexHandle
{
	SVertexHandle(uint64 ID)
	    : Handle(ID)
	{
	}

	SVertexHandle(const SVertexHandle& ID)
	    : Handle(ID.Handle)
	{
	}

	SVertexHandle() = default;

	NODISCARD uint64 GetHandle() const
	{
		return Handle;
	}

	explicit operator int64() const
	{
		return Handle;
	}

	SVertexHandle& operator=(const SVertexHandle& OtherHandle) = default;

	friend bool operator==(const SVertexHandle& FirstHandle, const SVertexHandle& SecondHandle)
	{
		return FirstHandle.Handle == SecondHandle.Handle;
	}

private:
	uint64 Handle = UINT64_MAX;
};

struct SDrawVertexHandle : SVertexHandle
{
	explicit SDrawVertexHandle(uint64 Handle)
	    : SVertexHandle(Handle) {}
	SDrawVertexHandle() = default;
};

struct SBufferAttribVertexHandle : SVertexHandle
{
	explicit SBufferAttribVertexHandle(uint64 Handle)
	    : SVertexHandle(Handle) {}
	SBufferAttribVertexHandle() = default;
};

struct SBufferHandle : SVertexHandle
{
	explicit SBufferHandle(uint64 Handle)
	    : SVertexHandle(Handle) {}
	SBufferHandle() = default;
};