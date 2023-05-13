#pragma once
#include "boost/multi_index/hashed_index.hpp"
#include "boost/multi_index/random_access_index.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index_container.hpp"

#include <boost/multi_index/member.hpp>

template<typename ValueType>
using LinkedHashSet = boost::multi_index_container<ValueType, boost::multi_index::sequenced<>>;
