#pragma once

#ifndef RENDERAPI_MODEL_HPP
#define RENDERAPI_MODEL_HPP

#include "Math.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"

/**@brief Model for storing */
struct SParsedModelContext
{
	OVector<OVec3> Vertices;
	OVector<OVec2> TexCoords;
	OVector<OVec3> Normals;
};

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

	void SetupModelFromModelContext(SParsedModelContext&& Context);
	void SetupModelFromModelContext(const SParsedModelContext& Context);

protected:
	OVector<int32> Indices;

	OVector<OVec3> Vertices;
	OVector<OVec2> TexCoords;
	OVector<OVec3> Normals;



};

} // namespace RAPI

#endif // RENDERAPI_MODEL_HPP
