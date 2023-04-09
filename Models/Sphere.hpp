
#ifndef RENDERAPI_SPHERE_HPP
#define RENDERAPI_SPHERE_HPP

#include "Math.hpp"
#include "Model.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"
namespace RAPI
{

class OSphere final : public OModel
{
	using Super = OModel;

public:
	OSphere()
	{
		PreInit(DefaultPrecision);
		Init(DefaultPrecision);
	}

	explicit OSphere(uint32 Precision)
	    : OModel(Precision)
	{
		PreInit(Precision);
		Init(Precision);
	}

	void Init(int32) noexcept override;
	void GetVertexTextureNormalPositions(SModelContext& OutContext) override;

private:
};

} // namespace RAPI

#endif // RENDERAPI_SPHERE_HPP
