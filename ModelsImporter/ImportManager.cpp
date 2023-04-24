//
// Created by Cybea on 4/17/2023.
//

#include "ImportManager.hpp"

namespace RAPI
{
OUniquePtr<OModel> OImportManager::BuildModelFromPath(const OPath& PathToModel, EModelType Type)
{
	IModelImporter* usedImporter = nullptr;
	switch (Type)
	{
	case EModelType::OBJ:
	{
		usedImporter = &OBJImporter;
		break;
	}
	}

	auto model = MakeUnique<OModel>();
	model->SetupModelFromModelContext(usedImporter->GetParsedModel(PathToModel));
	return Move(model);
}
} // namespace RAPI