#pragma once

template <typename FunctionType, typename... ArgTypes>
auto Execute(FunctionType &&Func, ArgTypes &&...Args) -> decltype(std::forward<FunctionType>(Func)(std::forward<ArgTypes>(Args)...))
{
    return std::forward<FunctionType>(Func)(std::forward<ArgTypes>(Args)...);
}

template <typename FunctionType, typename OwnerType, typename ... ArgTypes>
auto Execute(FunctionType &&Func,OwnerType* Owner, ArgTypes &&...Args) -> decltype(Owner->*Func(std::forward<ArgTypes>(Args)...))
{
    return Owner->*Func(std::forward<ArgTypes>(Args)...);
}