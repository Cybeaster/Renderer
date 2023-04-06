//
// Created by Cybea on 4/4/2023.
//

#include "InterruptibleThread.hpp"

#include "InterruptFlag.hpp"

namespace RenderAPI
{
NODISCARD bool OInterruptibleThread::IsJoinable() const
{
	InternalThread.joinable();
}
void OInterruptibleThread::Interrupt()
{
	if (Flag)
	{
		Flag->Set();
	}
}
void OInterruptibleThread::Detach()
{
	InternalThread.detach();
}
void OInterruptibleThread::Join()
{
	InternalThread.join();
}

} // namespace RenderAPI
