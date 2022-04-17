#pragma once
#include <Test.hpp>
#include <Texture.hpp>
class GLFWwindow;
namespace test
{
    class TestTexture : public Test
    {
        public:

        TestTexture(std::string filePath);
        ~TestTexture();

        void OnUpdate(GLFWwindow* window,float currentTime);
        
        private:
        
        Texture texture;
    };
    
  
} // namespace test
