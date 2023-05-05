
#include "Models/Model.hpp"
#include "SmartPtr.hpp"
namespace RAPI
{

class ODrawPool
{

private:

	OVector<OUniquePtr<OModel>> ModelsToDraw;
};

} // namespace RAPI
