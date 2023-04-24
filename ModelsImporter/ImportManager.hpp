#pragma once
#include "ModelsImporter/OBJImporter/ObjImporter.hpp"
namespace RAPI
{

enum class EModelType
{
	OBJ
};

class OImportManager
{
public:
	OUniquePtr<OModel> BuildModelFromPath(const OPath& PathToModel, EModelType Type);

private:
	OOBJImporter OBJImporter;
};

} // namespace RAPI
