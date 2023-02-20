#include "Application/Application.hpp"
#include "MulticastDelegate.hpp"


int main(int argc, char** argv)
{
	auto app = OApplication::GetApplication();
	app->Start(argc, argv);
	
	// object ob;
	// RenderAPI::OTMulticastDelegate<bool> delegate;


	
	// delegate.AddRaw(&ob, &object::print);
	// delegate.Broadcast(false);

	return 0;
}


// int main(int argc, char** argv)
// {
	// object ob;
	// RenderAPI::OTMulticastDelegate<bool> delegate;


	
	// delegate.AddRaw(&ob, &object::print);
	// delegate.Broadcast(false);

// 	return 0;
// }
