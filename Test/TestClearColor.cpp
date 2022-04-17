#include "TestClearColor.hpp"
#include "Renderer.hpp"
namespace test
{
    

     ClearColorTest::ClearColorTest( ) : m_Clearcolor{0.2f,0.3f,0.8f,1.f}{

     }

    ClearColorTest::~ClearColorTest()
    {

    }
        
    void ClearColorTest::OnRender() 
    {
        GLCall(glClearColor(m_Clearcolor[0],m_Clearcolor[1],m_Clearcolor[2],m_Clearcolor[3]));
        GLCall(glClear(GL_COLOR_BUFFER_BIT));

    }

    void ClearColorTest::OnUpdate(float deltaTime)
    {

    }

    void ClearColorTest::OnImGuiRender() 
    {
        
    }

} // namespace test
