#pragma once

#include "Threads/Thread.hpp"
#include "Threads/Utils.hpp"
#include "TypeTraits.hpp"

namespace RAPI
{

class OInterruptibleThread
{
public:
	template<typename FuncType>
	explicit OInterruptibleThread(FuncType Function);

	void Join();
	void Detach();
	void Interrupt();

	NODISCARD bool IsJoinable() const;

private:
	OThread InternalThread;
	OInterruptFlag* Flag;
};

template<typename FuncType>
OInterruptibleThread::OInterruptibleThread(FuncType Function)
{
	OPromise<OInterruptFlag*> pFlag;
	InternalThread = OThread([Function, &pFlag]
	                         {
		                         pFlag.set_value(&LocalThreadInterruptFlag);

		                         try
		                         {
			                         Function();
		                         }
		                         catch (const boost::thread_interrupted&)
		                         {
		                         }
	                         });

	Flag = pFlag.get_future().get();
}

} // namespace RAPI
