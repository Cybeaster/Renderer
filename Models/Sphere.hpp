
#ifndef RENDERAPI_SPHERE_HPP
#define RENDERAPI_SPHERE_HPP

#include "Math.hpp"
#include "Tuple.hpp"
#include "TypeTraits.hpp"
#include "Types.hpp"
#include "Vector.hpp"
namespace RenderAPI
{

class OSphere
{
public:
	OSphere();
	explicit OSphere(uint32 Precision);

	NODISCARD FORCEINLINE uint32 GetNumVertices() const
	{
		return NumVertices;
	}

	NODISCARD FORCEINLINE uint32 GetNumIndices() const
	{
		return NumIndices;
	}

	void GetVertexTextureNormalPositions(OTVector<float>& OutVertex, OTVector<float>& OutTexture, OTVector<float>& OutNormals);

private:
	void Init(int32) noexcept;

	uint32 NumVertices;
	uint32 NumIndices;

	const uint32 DefaultPrecision = 48;

	OTVector<int32> Indices;
	OTVector<OVec3> Vertices;
	OTVector<OVec2> TexCoords;
	OTVector<OVec3> Normals;
};

} // namespace RenderAPI

#endif // RENDERAPI_SPHERE_HPP
