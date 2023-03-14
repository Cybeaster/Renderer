#pragma once
#include "Application/Application.hpp"
#include "Utils/Threads/Tests/ThreadTests.hpp"

int main(int argc, char** argv)
{
	auto app = RenderAPI::OApplication::GetApplication();
	app->Start(argc, argv);

	return 0;
}