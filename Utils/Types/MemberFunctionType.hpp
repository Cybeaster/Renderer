#pragma once


template<typename ObjectType, typename RetType, typename ... ArgTypes>
struct MemberFunctionType
{
    using Type = RetType(ObjectType::*)(ArgTypes...);
};
