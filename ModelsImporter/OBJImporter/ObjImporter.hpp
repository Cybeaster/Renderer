#pragma once
#include "../ModelImporter.hpp"
namespace RAPI
{

class OOBJImporter : public IModelImporter
{
public:
private:
	NODISCARD SParsedModelContext GetParsedModel(const OPath& PathToModel) const override;
};

} // namespace RAPI
