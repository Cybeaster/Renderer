#include "Application/Application.hpp"




int main(int argc, char **argv)
{
    Application* app = Application::GetApplication();
	app->Start(argc,argv);
    return 0;
}