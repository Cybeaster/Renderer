#include "GeneratedModel.hpp"
#include "Model.hpp"

#ifndef RENDERAPI_TORUS_HPP
#define RENDERAPI_TORUS_HPP

namespace RAPI
{

class OTorus final : public OGeneratedModel
{
	using Super = OGeneratedModel;

public:
	OTorus()
	    : OGeneratedModel(DefaultPrecision)
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	explicit OTorus(uint32 Precision, float Inner, float Outer)
	    : OGeneratedModel(Precision), InnerRadius(Inner), OuterRadius(Outer)
	{
		PreInit(Precision);
		Init(Precision);
	}

	explicit OTorus(uint32 Precision)
	    : OGeneratedModel(Precision)
	{
		PreInit(Precision);
		Init(Precision);
	}

	void Init(uint32) noexcept override;
	void PreInit(uint32) noexcept override;
	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;
	SModelContext GetModelContext();

private:
	void CalcFirstRing(uint32 Precision);
	void MultiplyRings(uint32 Precision);
	void CalcIndices(uint32 Precision);

	float InnerRadius{ 0.5F };
	float OuterRadius{ 0.2F };

	OVector<OVec3> STangents;
	OVector<OVec3> TTangents;
};

} // namespace RAPI

#endif // RENDERAPI_TORUS_HPP
