#pragma once
#include "Types.hpp"

namespace RenderAPI
{

enum EDrawType
{
	// Draws vertices
	DrawArrays,
	// Draws indices
	DrawElements
};

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

	    DrawFlagType(Type)
	    , FirstDrawIndex(Index)
	    , DrawSize(Size)
	    , DepthFunction(Function)
	    , FrontFace(FrontFaceArg)
	    , Flag(FlagArg)
	{
	}

	uint32 DrawFlagType = UINT32_INVALID_VALUE;
	uint32 FirstDrawIndex = UINT32_INVALID_VALUE;
	uint32 DrawSize = UINT32_INVALID_VALUE;
	uint32 DepthFunction = UINT32_INVALID_VALUE;
	uint32 FrontFace = UINT32_INVALID_VALUE;
	uint32 Flag = UINT32_INVALID_VALUE;
};
} // namespace RenderAPI
