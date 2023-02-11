#pragma once
#include <Types.hpp>
class OBuffer
{
public:
	OBuffer(const void* Data, size_t size);

	OBuffer(const OBuffer& Buffer) = default;
	OBuffer& operator=(const OBuffer& Buffer) = default;
    
	OBuffer() = default;

	~OBuffer();

	void Bind() const;
	void Unbind() const;

private:
	uint32 BufferID;
};