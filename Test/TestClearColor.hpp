#include "Test.hpp"

namespace test
{
    
    class ClearColorTest : public Test
    {

    public:

        ClearColorTest(/* args */);
        virtual ~ClearColorTest() override;
        
        void OnRender() override;
        void OnUpdate(float deltaTime) override;
        void OnImGuiRender() override;

    private:
        float m_Clearcolor[4];

    };


} // namespace test
