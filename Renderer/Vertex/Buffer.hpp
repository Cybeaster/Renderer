#pragma once
#include <Types.hpp>

enum EBufferTypes
{
	ArrayBuffer = 0x8892,
	ElementArrayBuffer = 0x8893
};

enum EBufferOptions
{
	StaticDraw = 0x88E4,
};

struct SBufferContext
{
	SBufferContext() = default;
	SBufferContext(const SBufferContext&) = default;

	SBufferContext(const void* DataArg, const uint32 SizeArg, EBufferOptions OptionArg, EBufferTypes TypeArg)
	    : Data(DataArg), Size(SizeArg), Option(OptionArg), Type(TypeArg) {}

	const void* Data;
	uint32 Size;

	EBufferOptions Option;
	EBufferTypes Type;
};

class OBuffer
{
public:
	OBuffer(const void* Data, size_t size);

	explicit OBuffer(const SBufferContext& Context);

	OBuffer() = default;

	~OBuffer();

	OBuffer(const OBuffer& Buffer) = default;

	OBuffer& operator=(const OBuffer& Buffer) = default;
	bool operator==(const OBuffer& Buffer) const = default;

	void Bind() const;
	void Unbind() const;

private:
	void Init(const void* Data, size_t size);

	EBufferOptions BufferOption = StaticDraw;
	EBufferTypes BufferType = ArrayBuffer;

	uint32 BufferID;
};