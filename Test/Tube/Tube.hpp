#pragma once
#include <vector>
#include "Pipe.h"
namespace test
{
    
    class Tube
    {
      
        public:
            Tube();
            ~Tube();

        private:
        void createVertices();
        
        Pipe pipe;
    };

} // namespace test
