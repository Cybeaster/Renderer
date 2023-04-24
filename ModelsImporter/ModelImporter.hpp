#pragma once

#include "Models/Model.hpp"
#include "Path.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{

// Base class for importers
class IModelImporter
{
public:
	NODISCARD virtual SParsedModelContext GetParsedModel(const OPath& PathToModel) const = 0;
};

} // namespace RAPI