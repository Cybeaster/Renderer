#pragma once
#include "Model.hpp"
#include "Types.hpp"

namespace RAPI
{

class OGeneratedModel : public OModel
{
public:
	OGeneratedModel() = default;

	OGeneratedModel(uint32 Precision)
	    : DefaultPrecision(Precision)
	{
	}

	NODISCARD FORCEINLINE uint32 GetNumVertices() const
	{
		return NumVertices;
	}

	NODISCARD FORCEINLINE uint32 GetNumIndices() const
	{
		return NumIndices;
	}

protected:
	virtual void Init(uint32) noexcept = 0;
	virtual void PreInit(uint32 Precision) noexcept;

	const uint32 DefaultPrecision = 48;
	mutable uint32 SqredPrecision{ DefaultPrecision * DefaultPrecision };

	uint32 NumVertices{ UINT32_INVALID_VALUE };
	uint32 NumIndices{ UINT32_INVALID_VALUE };
};

} // namespace RAPI
