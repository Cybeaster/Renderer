
#include "Utils.hpp"

#include "Logging/Log.hpp"

namespace RenderAPI
{

OString OThreadUtils::GetFormattedThreadID()
{
	std::ostringstream ss;
	ss << NThisThread::GetThreadID();

	return SLogUtils::Format("Current Thread ID is {} \n ", ss.str());
}
} // namespace RenderAPI