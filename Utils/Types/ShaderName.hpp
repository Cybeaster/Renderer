#pragma once

#include "TypeTraits.hpp"
namespace RAPI
{

struct SShaderName
{
	SShaderName() = default;
	SShaderName(const SShaderName& Str) = default;

	explicit SShaderName(const OString& Str) // NOLINT
	    : Name(Str)
	{
	}

	SShaderName(OString&& Str) noexcept // NOLINT
	    : Name(Move(Str))
	{
	}

	SShaderName(const char* Str) // NOLINT
	    : Name(Str)
	{
	}

	SShaderName(SShaderName&& Str) noexcept
	    : Name(Move(Str)) {}

	SShaderName& operator=(const OString& Str)
	{
		Name = Str;
		return *this;
	}

	SShaderName& operator=(const char* const Str)
	{
		Name = Str;
		return *this;
	}

	explicit operator OString() const
	{
		return Name;
	}

	OString Name;
};

} // namespace RAPI