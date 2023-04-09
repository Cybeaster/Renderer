#include "InterruptFlag.hpp"

#include "../Utils.hpp"
namespace RAPI
{

OInterruptFlag::SClearCondVariableOnDestruct::~SClearCondVariableOnDestruct()
{
	LocalThreadInterruptFlag.ClearConditionVariable();
}

} // namespace RAPI
