#include "Application/Application.hpp"


int main(int argc, char** argv)
{
	auto app = OApplication::GetApplication();
	app->Start(argc, argv);

	return 0;
}