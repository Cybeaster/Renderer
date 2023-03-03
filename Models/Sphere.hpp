
#ifndef RENDERAPI_SPHERE_HPP
#define RENDERAPI_SPHERE_HPP

#include "Math.hpp"
#include "Model.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"
namespace RenderAPI
{

class OSphere final : public OModel
{
public:
	explicit OSphere(uint32 Precision)
	    : OModel(Precision) {}

	void Init(int32) noexcept override;
	
private:
};

} // namespace RenderAPI

#endif // RENDERAPI_SPHERE_HPP
