#pragma once

template <typename ObjectType, typename RetType, typename... ArgTypes>
struct TTMemberFunctionType
{
    using Type = RetType (ObjectType::*)(ArgTypes...);
};
