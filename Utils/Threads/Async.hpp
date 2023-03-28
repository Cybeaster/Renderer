#pragma once

#ifndef RENDERAPI_ASYNC_HPP
#define RENDERAPI_ASYNC_HPP

#include "TypeTraits.hpp"
#include "Utils/Types/Threads/Thread.hpp"

namespace Async
{
template<typename Func, typename Arg>
OFuture<std::invoke_result_t<Func(Arg&&)>> SpawnTask(Func&& Function, Arg&& Argument)
{
	using TResultType = std::invoke_result_t<Func(Arg &&)>;
	std::packaged_task<TResultType(Arg &&)> task(Move(Function));

	std::future<TResultType> result(task.get_future());
	std::thread newThread(Move(task), Move(Argument));

	newThread.detach();
	return result;
}

} // namespace Async

#endif // RENDERAPI_ASYNC_HPP
