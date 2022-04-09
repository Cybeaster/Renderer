
#pragma once

namespace test
{
    class Test
    {

    public:

        Test(/* args */);
        virtual ~Test();
        
        virtual void OnUpdate(float deltaTime) =0;
        virtual void OnRender() = 0;
        virtual void OnImGuiRender(){}

    private:


   

    };
    
 
}