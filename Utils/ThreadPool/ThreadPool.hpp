#include <RenderAPITypes.hpp>
#include <JoiningThread.hpp>
namespace RenderAPI
{
    namespace Thread
    {
        class ThreadPool
        {
        public:
            ThreadPool(/* args */);
            ~ThreadPool();

            void CreateThread(JoiningThread&& Thread);
        private:
            
            Vector<JoiningThread> Threads; 
        };

    } // namespace Thread

} // namespace RenderAPI
