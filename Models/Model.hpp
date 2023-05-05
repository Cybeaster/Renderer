#pragma once

#ifndef RENDERAPI_MODEL_HPP
#define RENDERAPI_MODEL_HPP

#include "Math.hpp"
#include "Texture.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"

/**@brief Ready to use in graphics API*/
struct SModelContext
{
	OVector<float> Vertices;
	OVector<float> TexCoords;
	OVector<float> Normals;
};

namespace RAPI
{

class OModel
{
public:
	OModel()
	{
	}

	explicit OModel(const uint32 /*Precision*/)
	{
	}

	NODISCARD FORCEINLINE const OVector<int32>& GetIndices() const
	{
		return Indices;
	}

	virtual void GetVertexTextureNormalPositions(SModelContext& OutContext);

protected:
	OVector<int32> Indices;
	OVector<OVec3> Vertices;
	OVector<OVec2> TexCoords;
	OVector<OVec3> Normals;

	OTexture* ModelTexture;
	OVec3 ModelPosition;
};

} // namespace RAPI

#endif // RENDERAPI_MODEL_HPP
