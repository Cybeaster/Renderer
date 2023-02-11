#pragma once
#include <Types.hpp>

struct TVertexHandle
{
	TVertexHandle(uint64 ID)
	    : Handle(ID)
	{
	}

	TVertexHandle(const TVertexHandle& ID)
	    : Handle(ID.Handle)
	{
	}

	TVertexHandle() = default;

	uint64 GetHandle() const
	{
		return Handle;
	}

	operator int64()
	{
		return Handle;
	}

	bool operator==(const TVertexHandle& ArrayHandle)
	{
		return Handle == ArrayHandle.Handle;
	}

	bool operator!=(const TVertexHandle& ArrayHandle)
	{
		return Handle != ArrayHandle.Handle;
	}

	bool operator>(const TVertexHandle& ArrayHandle)
	{
		return Handle > ArrayHandle.Handle;
	}

	bool operator>=(const TVertexHandle& ArrayHandle)
	{
		return Handle >= ArrayHandle.Handle;
	}

	bool operator<(const TVertexHandle& ArrayHandle)
	{
		return Handle < ArrayHandle.Handle;
	}

	bool operator<=(const TVertexHandle& ArrayHandle)
	{
		return Handle <= ArrayHandle.Handle;
	}

	TVertexHandle& operator=(const TVertexHandle& OtherHandle)
	{
		Handle = OtherHandle.Handle;
		return *this;
	}

	friend bool operator==(const TVertexHandle& FirstHandle, const TVertexHandle& SecondHandle)
	{
		return FirstHandle.Handle == SecondHandle.Handle;
	}

private:
	uint64 Handle = UINT64_MAX;
};

struct TDrawVertexHandle : TVertexHandle
{
	explicit TDrawVertexHandle(uint64 Handle)
	    : TVertexHandle(Handle) {}
	TDrawVertexHandle() = default;
};

struct OBufferAttribVertexHandle : TVertexHandle
{
	explicit OBufferAttribVertexHandle(uint64 Handle)
	    : TVertexHandle(Handle) {}
	OBufferAttribVertexHandle() = default;
};