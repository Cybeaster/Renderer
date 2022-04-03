
#pragma once

namespace test
{
    

    class Test
    {

    public:

        Test(/* args */);
        virtual ~Test();
        
        virtual void OnUpdate(float deltaTime){}
        virtual void OnRender(){}
        virtual void OnImGuiRendere() {}

    private:


   

    };
    
 
}