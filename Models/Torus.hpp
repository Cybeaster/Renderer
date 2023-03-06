#include "Model.hpp"

#ifndef RENDERAPI_TORUS_HPP
#define RENDERAPI_TORUS_HPP

namespace RenderAPI
{

class OTorus final : public OModel
{
	using Super = OModel;

public:
	OTorus()
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	explicit OTorus(uint32 Precision, float Inner, float Outer)
	    : OModel(Precision), InnerRadius(Inner), OuterRadius(Outer)
	{
		PreInit(Precision);
		Init(Precision);
	}

	explicit OTorus(uint32 Precision)
	    : OModel(Precision)
	{
		PreInit(Precision);
		Init(Precision);
	}

	void Init(int32) noexcept override;
	void PreInit(int32) noexcept override;
	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;

private:
	void CalcFirstRing(uint32 Precision);
	void MultiplyRings(uint32 Precision);
	void CalcIndices(uint32 Precision);

	float InnerRadius{ 0.5F };
	float OuterRadius{ 0.2F };

	OTVector<OVec3> STangents;
	OTVector<OVec3> TTangents;
};

} // namespace RenderAPI

#endif // RENDERAPI_TORUS_HPP
