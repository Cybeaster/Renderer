#include "InterruptFlag.hpp"

#include "../Utils.hpp"
namespace RenderAPI
{

OInterruptFlag::SClearCondVariableOnDestruct::~SClearCondVariableOnDestruct()
{
	OThreadUtils::LocalThreadInterruptFlag.ClearConditionVariable();
}
} // namespace RenderAPI
