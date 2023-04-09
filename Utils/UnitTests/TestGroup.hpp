#pragma once

#ifndef RENDERAPI_TESTGROUP_HPP
#define RENDERAPI_TESTGROUP_HPP

#include "../SmartPtr.hpp"
#include "Logging/Log.hpp"
#include "Vector.hpp"

namespace RAPI
{

class OTestBase
{
public:
	NODISCARD constexpr virtual OString GetName() const = 0;

	virtual void Run()
	{
		RAPI_LOG(Log, "Test {} is running!", GetName());
	}
	virtual void PostRun()
	{
		RAPI_LOG(Log, "Test {} is stopped!", GetName());
	}
};

class OTestGroup
{
public:
	static auto Get()
	{
		if (Group == nullptr)
		{
			Group = new OTestGroup();
		}
		return Group;
	}

	void AddTest(OTestBase* Test)
	{
		Tests.push_back(Test);
	}

	void Run()
	{
		for (auto& test : Tests)
		{
			test->Run();
		}
	}

	void Run(OString&& TestName)
	{
		for (auto& test : Tests)
		{
			if (test->GetName() == TestName)
			{
				test->Run();
				return;
			}
		}
	}

private:
	OTestGroup() = default;

	static inline OTestGroup* Group{ nullptr };
	OVector<OTestBase*> Tests{};
};

} // namespace RAPI

#define MAKE_TEST(Name)                             \
	struct S##Name##Test : public RAPI::OTestBase   \
	{                                               \
		using ThisTestType = S##Name##Test;         \
		using Super = RAPI::OTestBase;              \
                                                    \
		S##Name##Test()                             \
		{                                           \
			RAPI::OTestGroup::Get()->AddTest(this); \
		}                                           \
                                                    \
		NODISCARD OString GetName() const override  \
		{                                           \
			return #Name;                           \
		}                                           \
                                                    \
		void Run() override;                        \
	} Name##Test;

#define RUN_ALL_TESTS() \
	RAPI::OTestGroup::Get()->Run();

#define RUN_TEST(TestName) \
	RAPI::OTestGroup::Get()->Run(Move(#TestName));

#endif // RENDERAPI_TESTGROUP_HPP
