//
// Created by Cybea on 4/16/2023.
//
#include "ObjImporter.hpp"

#include "Assert.hpp"

#include <fstream>
#include <sstream>

namespace RAPI
{
SModelContext OOBJImporter::GetParsedModel(const OPath& PathToModel) const
{
	SModelContext result;

	OVector<float> vertices;
	OVector<float> texels;
	OVector<float> normals;

	float x, y, z;

	OString content;

	std::ifstream fileStream(PathToModel.string(), std::ios::in); // TODO make custom file stream

	OString line;
	ASSERT(fileStream.is_open());

	while (std::getline(fileStream, line))
	{
		// vertex pos
		if (line.compare(0, 2, "v ") == 0)
		{
			std::stringstream ss(line.erase(0, 1));
			ss >> x;
			ss >> y;
			ss >> z;
			vertices.push_back(x);
			vertices.push_back(y);
			vertices.push_back(z);
		}

		// texture coord
		if (line.compare(0, 2, "vt") == 0)
		{
			std::stringstream ss(line.erase(0, 2));

			ss >> x;
			ss >> y;

			texels.push_back(x);
			texels.push_back(y);
		}

		// Check normals
		if (line.compare(0, 2, "vn") == 0)
		{
			std::stringstream ss(line.erase(0, 2));
			ss >> x;
			ss >> y;
			ss >> z;

			normals.push_back(x);
			normals.push_back(y);
			normals.push_back(z);
		}

		// Check faces
		if (line.compare(0, 1, "f") == 0)
		{
			OString oneCorner, vertex, texel, normal;
			std::stringstream ss(line.erase(0, 2));

			for (uint32 it = 0; it < 3; ++it)
			{
				std::getline(ss, oneCorner, ' ');

				std::stringstream oneCornerSS(oneCorner);

				std::getline(oneCornerSS, vertex, '/');
				std::getline(oneCornerSS, texel, '/');
				std::getline(oneCornerSS, normal, '/');

				// subtract to make it 0 based
				auto vertRef = (std::stoi(vertex) - 1) * 3;
				auto texelRef = ((std::stoi(texel)) - 1) * 2;
				auto normRef = ((std::stoi(normal)) - 1) * 3;

				result.Vertices.push_back(vertices[vertRef]);
				result.Vertices.push_back(vertices[vertRef + 1]);
				result.Vertices.push_back(vertices[vertRef + 2]);

				result.TexCoords.push_back(texels[texelRef]);
				result.TexCoords.push_back(texels[texelRef + 1]);

				result.Normals.push_back(normals[normRef]);
				result.Normals.push_back(normals[normRef + 1]);
				result.Normals.push_back(normals[normRef + 2]);
			}
		}
		RAPI_LOG(Log, "Loading model {} ...", PathToModel.string());
	}

	RAPI_LOG(Log, "Model {} has been loaded!", PathToModel.string());
	return result;
}
} // namespace RAPI