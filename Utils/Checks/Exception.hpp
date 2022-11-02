#pragma once
#include <exception>

namespace RenderAPI
{
    namespace Exceptions
    {
        class TException : std::exception
        {
        public:
            explicit TException(const char* Msg) : Message(Msg){}
            explicit TException(const TString& Msg) : Message(std::move(Msg)){};

            virtual ~TException(){}

            virtual const char* what()const noexcept
            {
                return Message.c_str();
            }
        private:
            TString Message;
       
        };
        
    } // namespace Exceptions
    
} // namespace RenderAPI
