#pragma once
#include <Test.hpp>
#include <Texture.hpp>
namespace test
{
    class TestTexture : public Test
    {
        public:

        TestTexture(std::string filePath);
        ~TestTexture();

        void OnUpdate(float deltaTime) override;
        void OnRender()override;
        void OnImGuiRender() override;
        
        private:
        
        Texture texture;
    };
    
  
} // namespace test
