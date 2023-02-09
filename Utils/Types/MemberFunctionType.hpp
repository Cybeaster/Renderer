#pragma once

template<typename ObjectType, typename RetType, typename... ArgTypes>
struct STMemberFunctionType
{
	using TConstFunction = RetType (ObjectType::*)(ArgTypes...) const;
	using TFunction = RetType (ObjectType::*)(ArgTypes...);
};
