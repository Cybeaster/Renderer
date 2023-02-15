#pragma once

// template<typename ObjectType, typename RetType, typename... ArgTypes>
// struct STMemberFunctionType
// {
// 	using TConstFunction = RetType (ObjectType::*)(ArgTypes...) const;
// 	using TFunction = RetType (ObjectType::*)(ArgTypes...);
// };

template<bool IsConst, typename ObjectType, typename RetType, typename... ArgTypes>
struct STMemberFunctionType
{
};

template<typename ObjectType, typename RetType, typename... ArgTypes>
struct STMemberFunctionType<false, ObjectType, RetType, ArgTypes...>
{
	using Type = RetType (ObjectType::*)(ArgTypes...);
};

template<typename ObjectType, typename RetType, typename... ArgTypes>
struct STMemberFunctionType<true, ObjectType, RetType, ArgTypes...>
{
	using Type = RetType (ObjectType::*)(ArgTypes...) const;
};
