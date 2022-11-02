#pragma once
#include <unordered_map>
#include <map>

template<typename Key,typename Value>
using TTHashMap = std::unordered_map<Key,Value>;

template<typename Key,typename Value>
using TTUniqueHashMap = std::map<Key,Value>;