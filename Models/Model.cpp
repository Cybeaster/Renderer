#include "Model.hpp"

#include "Assert.hpp"
#include "Logging/Log.hpp"
namespace RAPI
{

void OModel::GetVertexTextureNormalPositions(SModelContext& OutContext)
{
	auto triangleVert = Vertices.size();

	OutContext.Vertices.resize(Vertices.size() * 3);
	OutContext.TexCoords.resize(TexCoords.size() * 2);
	OutContext.Normals.resize(Normals.size() * 3);

	for (int32 it = 0; it < triangleVert; it++)
	{
		OutContext.Vertices[it * 3] = Vertices[it].x;
		OutContext.Vertices[it * 3 + 1] = Vertices[it].y;
		OutContext.Vertices[it * 3 + 2] = Vertices[it].z;

		OutContext.TexCoords[it * 2] = TexCoords[it].t;
		OutContext.TexCoords[it * 2 + 1] = TexCoords[it].s;

		OutContext.Normals[it * 3] = Normals[it].x;
		OutContext.Normals[(it * 3) + 1] = Normals[it].y;
		OutContext.Normals[(it * 3) + 2] = Normals[it].z;
	}
}

} // namespace RAPI