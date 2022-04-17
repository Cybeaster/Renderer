#pragma once
#include "GL/glew.h"
#include "glfw3.h"
namespace test
{

    class Test
    {

    public:

        Test(/* args */);
        virtual ~Test();
        
        virtual void OnUpdate(GLFWwindow* window, float deltaTime) = 0;

    private:

    };
    
 
}