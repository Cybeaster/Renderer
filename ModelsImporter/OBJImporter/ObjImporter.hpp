#pragma once
#include "../ModelImporter.hpp"
namespace RAPI
{

class OOBJImporter : public IModelImporter
{
public:
private:
	NODISCARD SModelContext GetParsedModel(const OPath& PathToModel) const override;
};

} // namespace RAPI
