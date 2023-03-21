//
// Created by Cybea on 3/7/2023.
//
#include "Types.hpp"
#include "Utils/UnitTests/TestGroup.hpp"

#include <barrier>
#include <cassert>
#include <future>

namespace Test
{

namespace Zero
{
struct X
{
	void foo(int, OString string);
};
bool expected = false;

std::atomic<bool> b{};

class Obj
{
};

std::shared_ptr<Obj> someObject;

void process_global_data()
{
	std::shared_ptr<Obj> loc = std::atomic_load(&someObject);
	// process_local(loc);
}
void update_global_data()
{
	std::shared_ptr<Obj> local(new Obj);
	std::atomic_store(&someObject, local);
}

void fu()
{
	while (!b.compare_exchange_weak(expected, true) && !expected)
		;

	std::atomic<bool> newB;
	b.compare_exchange_weak(expected, true, std::memory_order::memory_order_acq_rel, std::memory_order::memory_order_acquire);

	Obj someObjects[5];

	std::atomic<Obj*> atom(someObjects);

	Obj* x = atom.fetch_add(2);
	assert(x = someObjects); // stays the same

	assert(atom.load() == &someObjects[2]);

	x = (atom -= 1);
	assert(x == &someObjects[2]);
	assert(atom.load() == &someObjects[1]);
}
} // namespace Zero

namespace One
{

std::vector<int> data;

std::atomic<bool> data_is_ready(false);

void reader_thread()
{
	while (!data_is_ready.load())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::cout << "The answer = " << data[0] << '\n';
}

void WriterThread()
{
	data.push_back(42);
	data_is_ready = true;
}
} // namespace One

namespace Two // memory_order_seq_cst
{
std::atomic<bool> x, y;
std::atomic<int> z;

void write_x()
{
	x.store(true, std::memory_order_seq_cst);
}

void write_y()
{
	y.store(true, std::memory_order_seq_cst);
}

void read_x_then_y()
{
	while (!x.load(std::memory_order::memory_order_seq_cst))
		;

	if (y.load(std::memory_order_seq_cst))
	{
		++z;
	}
}

void read_y_then_x()
{
	while (!y.load(std::memory_order::memory_order_seq_cst))
		;

	if (x.load(std::memory_order_seq_cst))
	{
		++z;
	}
}
int Testmain()
{
	x = false;
	y = false;

	z = 0;
	std::thread a(write_x);
	std::thread b(write_y);

	std::thread c(read_x_then_y);
	std::thread d(read_y_then_x);

	a.join();
	b.join();
	c.join();
	d.join();

	assert(z.load() != 0);
}
} // namespace Two

namespace Three
{
std::atomic<bool> x, y;
std::atomic<int> z;

void write_x_then_y()
{
	x.store(true, std::memory_order_relaxed);
	y.store(true, std::memory_order_relaxed);
}

void read_y_then_x()
{
	while (!y.load(std::memory_order::memory_order_relaxed))
		;

	if (x.load(std::memory_order_relaxed))
	{
		++z;
	}
}

int Testmain()
{
	x = false;
	y = false;

	z = 0;
	std::thread a(write_x_then_y);
	std::thread b(read_y_then_x);

	a.join();
	b.join();

	assert(z.load() != 0);
}
} // namespace Three

namespace Four
{
std::atomic<int> x(0), y(0), z(0);
std::atomic<bool> go(false);
const uint8 loop_count = 10;

struct ReadValues
{
	int x, y, z;
};

ReadValues values1[loop_count];
ReadValues values2[loop_count];
ReadValues values3[loop_count];
ReadValues values4[loop_count];
ReadValues values5[loop_count];

inline void Increment(std::atomic<int>* var_to_inc, ReadValues* Values)
{
	while (!go)
	{
		std::this_thread::yield();
	}

	for (int i = 0; i < loop_count; ++i)
	{
		Values[i].x = x.load(std::memory_order_seq_cst);
		Values[i].y = y.load(std::memory_order_seq_cst);
		Values[i].z = z.load(std::memory_order_seq_cst);

		var_to_inc->store(i + 1, std::memory_order_seq_cst);
		std::this_thread::yield();
	}
}

inline void read_vals(ReadValues* values)
{
	while (!go)
	{
		std::this_thread::yield();
	}

	for (uint32 i = 0; i < loop_count; ++i)
	{
		values[i].x = x.load(std::memory_order_seq_cst);
		values[i].y = y.load(std::memory_order_seq_cst);
		values[i].z = z.load(std::memory_order_seq_cst);
		std::this_thread::yield();
	}
}

inline void print_vals(ReadValues* v)
{
	for (int i = 0; i < loop_count; ++i)
	{
		if (i)
			std::cout << ",";
		std::cout << "(" << v[i].x << "," << v[i].y << "," << v[i].z << ")";
	}
	std::cout << std::endl;
}

void mainTest()
{
	std::thread t1(Increment, &x, values1);
	std::thread t2(Increment, &y, values2);
	std::thread t3(Increment, &z, values3);
	std::thread t4(read_vals, values4);
	std::thread t5(read_vals, values5);

	go = true;

	t5.join();
	t4.join();
	t3.join();
	t2.join();
	t1.join();

	print_vals(values1);
	print_vals(values2);
	print_vals(values3);
	print_vals(values4);
	print_vals(values5);
};

} // namespace Four

namespace Five
{
std::atomic<bool> x, y;
std::atomic<int> z;

void write_x()
{
	x.store(true, std::memory_order_release);
}

void write_y()
{
	y.store(true, std::memory_order_release);
}

void read_x_then_y()
{
	while (!x.load(std::memory_order::memory_order_acquire))
		;

	if (y.load(std::memory_order_acquire))
	{
		++z;
	}
}

void read_y_then_x()
{
	while (!y.load(std::memory_order::memory_order_acquire))
		;

	if (x.load(std::memory_order_acquire))
	{
		++z;
	}
}
void Test()
{
	x = false;
	y = false;

	z = 0;
	std::thread a(write_x);
	std::thread b(write_y);

	std::thread c(read_x_then_y);
	std::thread d(read_y_then_x);

	a.join();
	b.join();
	c.join();
	d.join();

	assert(z.load() != 0);
}
} // namespace Five

namespace Six
{
struct X
{
	int i;
	std::string s;
};

std::atomic<X*> p;
std::atomic<int> a;

void create_x()
{
	X* x = new X;
	x->i = 42;
	x->s = "hello";
	a.store(99, std::memory_order_relaxed);
	p.store(x, std::memory_order_release);
}

void use_x()
{
	X* x;
	while (!(x = p.load(std::memory_order_consume)))
	{
		std::this_thread::sleep_for(std::chrono::microseconds(1));
	}
	assert(x->i == 42);
	assert(x->s == "hello");
	assert(a.load(std::memory_order_relaxed) == 99);
}

void Run()
{
	std::thread t1(create_x);
	std::thread t2(use_x);
	t1.join();
	t2.join();
}

} // namespace Six

namespace Seven
{
std::vector<int> queueData;
std::atomic<int> count;

void populate_queue()
{
	unsigned const number_ofItems = 20;
	queueData.clear();
	for (int i = 0; i < number_ofItems; i++)
	{
		queueData.push_back(i);
	}
	count.store(number_ofItems, std::memory_order_release);
};

void consume_queue_items()
{
	while (true)
	{
		int item_index;
		if ((item_index = count.fetch_sub(1, std::memory_order_acquire)) <= 0)
		{
			// wait for more data;
			continue;
		}
		// process data;
	}
}
} // namespace Seven

MAKE_TEST(FenceDebuggable)
void SFenceDebuggableTest::Run()
{
	Super::Run();
	return;
	std::atomic<bool> x, y;
	std::atomic<int> z;

	auto write_x_then_y = [&]()
	{
		x.store(true, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		y.store(true, std::memory_order_relaxed);
	};

	auto read_y_then_x = [&]()
	{
		while (!y.load(std::memory_order_relaxed))
			;
		std::atomic_thread_fence(std::memory_order_acquire);
		if (x.load(std::memory_order_relaxed))
		{
			z++;
		}
	};

	x = false;
	y = false;
	z = 0;

	std::thread a(write_x_then_y);
	std::thread b(read_y_then_x);

	a.join();
	b.join();
	std::mutex mutex;
	mutex.lock();
	Super::PostRun();
}

MAKE_TEST(SharedPtr)
void SSharedPtrTest::Run()
{
	std::atomic<std::shared_ptr<int>> atomic;

	if (atomic.is_lock_free())
	{
		RAPI_LOG(Warning, "Shared ptr is lock free!")
	}
	else
	{
		RAPI_LOG(Error, "Shared ptr is not lock free!")
	}
}

} // namespace Test