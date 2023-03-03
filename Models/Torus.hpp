#include "Model.hpp"

#ifndef RENDERAPI_TORUS_HPP
#define RENDERAPI_TORUS_HPP

namespace RenderAPI
{

class OTorus : public OModel
{
public:
	OTorus() = default;

	explicit OTorus(uint32 Precision)
	    : OModel(Precision)
	{
	}

	void Init(int32) noexcept override;
};

} // namespace RenderAPI

#endif // RENDERAPI_TORUS_HPP
