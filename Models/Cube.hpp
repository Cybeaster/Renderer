
#pragma once
#include "GeneratedModel.hpp"
namespace RAPI
{

class OCube : public OGeneratedModel
{
public:
	OCube()
	    : OGeneratedModel()
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	void Init(uint32 Precision) noexcept override;
	void PreInit(uint32 Precision) noexcept override;

	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;

private:


};

} // namespace RAPI