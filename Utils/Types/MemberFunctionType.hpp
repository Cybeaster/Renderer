#pragma once

template <typename ObjectType, typename RetType, typename... ArgTypes>
struct TTMemberFunctionType
{
    using TConstFunction = RetType (ObjectType::*)(ArgTypes...) const;
    using TFunction = RetType (ObjectType::*)(ArgTypes...);
};
