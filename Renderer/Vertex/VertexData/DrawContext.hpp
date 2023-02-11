#pragma once
#include "Types.hpp"

namespace RenderAPI
{

struct SDrawFlag
{
	SDrawFlag() = default;

	explicit SDrawFlag(uint32 Arg)
	{
		Flag = Arg;
	}

	explicit operator const uint32&() const noexcept
	{
		return Flag;
	}

	uint32 Flag = UINT32_MAX;
};

struct SDrawContext
{
	SDrawContext(const SDrawContext& Context) = default;

	SDrawContext() = default;

	SDrawContext(const uint32 Type,
	             const uint32 Index,
	             const uint32 Size,
	             const uint32 Function,
	             const uint32 FrontFaceArg,
	             const uint32 FlagArg)
	    :

	    DrawType(Type)
	    , FirstDrawIndex(Index)
	    , DrawSize(Size)
	    , DepthFunction(Function)
	    , FrontFace(FrontFaceArg)
	    , Flag(FlagArg)
	{
	}

	uint32 DrawType = UINT32_MAX;
	uint32 FirstDrawIndex = UINT32_MAX;
	uint32 DrawSize = UINT32_MAX;
	uint32 DepthFunction = UINT32_MAX;
	uint32 FrontFace = UINT32_MAX;
	uint32 Flag = UINT32_MAX;
};
} // namespace RenderAPI
