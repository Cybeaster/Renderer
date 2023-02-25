#pragma once
#include "Buffer.hpp"
#include "VertexData/VertexContext.hpp"

#include <SmartPtr.hpp>
#include <Types.hpp>

namespace RenderAPI
{
class OVertexAttribBuffer
{
public:
	OVertexAttribBuffer(/* args */) = default;

	explicit OVertexAttribBuffer(const SVertexContext& Context);

	void EnableVertexAttribPointer();

private:
	SVertexContext VertexContext;
};
} // namespace RenderAPI
