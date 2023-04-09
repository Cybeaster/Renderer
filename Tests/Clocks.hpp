

#include <chrono>
#include <future>
#include <iostream>
struct SClocks
{
	SClocks()
	{
		{
			auto time = std::chrono::system_clock::now();

			using namespace std::chrono_literals;
			auto one_day = 24h;
			auto half_an_hour = 30min;
			auto max_time_between_message = 30ms;

			std::chrono::milliseconds ms(54264);
			std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(ms); // truncated!

			int sometask();
			std::future<int> future = std::async(sometask);
			if (future.wait_for(std::chrono::milliseconds(35)) == std::future_status::ready)
			{
				// do smth with future::get();
			}
		}

		{
			auto start = std::chrono::high_resolution_clock::now();
			// do smth();
			auto stop = std::chrono::high_resolution_clock::now();
			auto delay = start - stop;

			auto duration = std::chrono::duration<double, std::chrono::seconds::period>(delay);

			std::cout
			    << "do smth took"
			    << duration.count()
			    << "seconds" << std::endl;
		}

		{
			wait_loop();
		}

		{
			;
			std::this_thread::sleep_for(std::chrono::seconds(10));

			std::this_thread::sleep_until(std::chrono::high_resolution_clock::now() + std::chrono::seconds(10));

			std::timed_mutex mx;
			mx.try_lock_for(std::chrono::seconds(10));
		}
	}

	std::mutex m;
	bool isDone = false;
	std::condition_variable cv;
	bool wait_loop()
	{
		auto const time_out = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
		std::unique_lock<std::mutex> lock(m);

		while (!isDone)
		{
			if (cv.wait_until(lock, time_out) == std::cv_status::timeout)
			{
				break;
			}
		}
		return isDone;
	}
};