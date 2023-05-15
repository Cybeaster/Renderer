#pragma once
#include "GeneratedModel.hpp"

namespace RAPI
{

class OPlane : public OGeneratedModel
{
public:
	OPlane()
	    : OGeneratedModel()
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	void Init(uint32 Precision) noexcept override;
	void PreInit(uint32 Precision) noexcept override;

	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;
};

} // namespace RAPI