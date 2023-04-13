#pragma once

#include "Functor/Functor.hpp"
#include "HashMap/Hash.hpp"
#include "Pair.hpp"
#include "Queue.hpp"
#include "Threads/Thread.hpp"
#include "Types.hpp"
#include "Vector.hpp"
namespace RAPI
{

enum class EThreadID : uint8
{
	MainThread,
	RenderThread,
	PhysicsThread,
	InputThread
};

class ONamedThreadPool
{
	const uint8 NamedThreadsCount = 4;

	using TFunctor = std::function<void()>;
	using TQueue = OThreadSafeQueue<TFunctor>;

	using TNamedThread = OPair<OThread, OUniquePtr<TQueue>>;

public:
	ONamedThreadPool();

	template<typename TFunction>
	void AddTaskToThread(EThreadID ID, TFunction&& Task);

private:
	void Run();
	void InitThreads();

	OHashMap<EThreadID, TNamedThread> Threads;

	OMutex ThreadsMutex;
	// OVector<TNamedThread> Threads;
};

template<typename TFunction>
void ONamedThreadPool::AddTaskToThread(EThreadID ID, TFunction&& Task)
{

}
} // namespace RAPI
