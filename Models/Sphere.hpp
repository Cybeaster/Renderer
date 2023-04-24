
#ifndef RENDERAPI_SPHERE_HPP
#define RENDERAPI_SPHERE_HPP

#include "GeneratedModel.hpp"
#include "Math.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"
namespace RAPI
{

class OSphere final : public OGeneratedModel
{
	using Super = OGeneratedModel;

public:
	OSphere()
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	explicit OSphere(uint32 Precision)
	    : OGeneratedModel(Precision)
	{
		PreInit(Precision);
		Init(Precision);
	}

	void Init(uint32) noexcept override;
	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;

private:
};

} // namespace RAPI

#endif // RENDERAPI_SPHERE_HPP
