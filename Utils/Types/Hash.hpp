#pragma once
#include <unordered_map>
#include <map>

template<typename Key,typename Value, class Hasher = std::hash<Key>>
using OTHashMap = std::unordered_map<Key,Value,Hasher>;

template<typename Key,typename Value>
using OTUniqueHashMap = std::map<Key,Value>;


template<typename HashType>
auto GetHash(const HashType& Type)
{
    return std::hash<HashType>::_Do_hash(Type);
}