#pragma once
#include <tuple>
#include "Tuple/Tuple.hpp"
namespace RenderAPI
{
    // Begin
    template <typename... ArgTypes>
    class TSimpleTupleExp;

    // End
    template <>
    struct TSimpleTupleExp<>;

    template <typename Head, typename... Tail>
    class TSimpleTupleExp<Head, Tail...> : TSimpleTupleExp<Tail...>
    {
        template <size_t Index, typename Head, typename... ArgTypes>
        struct Getter
        {
            using ReturnType = typename Getter<Index - 1, ArgTypes...>::ReturnType;
            static ReturnType Get(TSimpleTupleExp<Head, ArgTypes...> Tuple)
            {
                return Getter<Index - 1, ArgTypes...>::Get(Tuple);
            }
        };

        template <typename Head, typename... Tail>
        struct Getter<0, Head, Tail...>
        {
            using ReturnType = typename TSimpleTupleExp<Head, Tail...>::ReturnType;
            static ReturnType Get(TSimpleTupleExp<Head, Tail...> T)
            {
                return T.Head;
            }
        };

    public:
        using BaseType = TSimpleTupleExp<Tail...>;
        using ValueType = Head;

        TSimpleTupleExp(Head _head, Tail... _tail) : head(_head), TSimpleTupleExp<Tail...>(_tail...) {}

        template <size_t Index, typename Head, typename... Tail>
        typename Getter<Index, Head, Tail...>::ReturnType Get()
        {
            return Getter<Index, Head, Args...>::Get(this);
        }

    private:
        // BaseType &baseType = static_cast<BaseType&>(this);
        ValueType head;
    };

#pragma endregion Tuple

    template <typename... Types>
    using TTuple = TTElemSequenceTuple<Types...>;
} // namespace RenderAPI
