#pragma once
#include "Buffer.hpp"
#include "VertexData/VertexContext.hpp"

#include <SmartPtr.hpp>
#include <Types.hpp>

namespace RenderAPI
{
class TVertexAttribBuffer
{
public:
	TVertexAttribBuffer(/* args */) = default;

	TVertexAttribBuffer(const SVertexContext& Context);

	void EnableVertexAttribPointer();

private:
	SVertexContext VertexContext;
};
} // namespace RenderAPI
