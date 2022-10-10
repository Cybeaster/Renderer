#include "ThreadPool.hpp"

namespace RenderAPI
{

    namespace Thread
    {
        ThreadPool::ThreadPool(/* args */)
        {
        }

        ThreadPool::~ThreadPool()
        {
        }

        void ThreadPool::CreateThread(JoiningThread &&Thread)
        {
            Threads.push_back(std::move(Thread));
        }
    } // namespace Thread

} // namespace RenderAPI
