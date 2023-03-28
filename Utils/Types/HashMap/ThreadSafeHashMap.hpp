#pragma once

#include "Hash.hpp"
#include "List.hpp"
#include "Pair.hpp"
#include "SmartPtr.hpp"
#include "Utils/Types/Threads/Thread.hpp"
#include "Vector.hpp"

#include <functional>
namespace RenderAPI
{

template<typename TKey, typename TValue, typename Hash = std::hash<TKey>>
class OThreadSafeHashMap
{
	class OBucket
	{
		friend OThreadSafeHashMap;
		using BucketEntityType = OPair<TKey, TValue>;
		using BucketData = OList<BucketEntityType>;
		using BucketIterator = typename BucketData::iterator;

	public:
		// Shared lock for reading
		TValue GetValueFor(const TKey& Key, const TValue& DefaultValue = TValue()) const
		{
			OSharedMutexLock lock(Mutex);
			BucketIterator entry = FindEntryFor(Key);
			return (entry == Data.end()) ? DefaultValue : entry->second;
		}

		// Unique lock for modifying
		void AddOrUpdate(const TKey& Key, const TValue& Value)
		{
			OUniqueLock<OSharedMutex> lock(Mutex);
			BucketIterator entry = FindEntryFor(Key);
			if (entry == Data.end())
			{
				Data.push_back(BucketEntityType(Key, Value));
			}
			else
			{
				entry->second = Value;
			}
		}

		bool Remove(const TKey& Key)
		{
			OUniqueLock<OSharedMutex> lock(Mutex);
			BucketIterator entry = FindEntryFor(Key);
			if (entry == Data.end())
			{
				return false;
			}

			Data.erase(entry);
			return true;
		}

	private:
		BucketIterator FindEntryFor(const TKey& Key) const
		{
			return std::find_if(Data.begin(), Data.end(), [&](const BucketEntityType& Item)
			                    { return Item.first == Key; });
		}

		BucketData Data;
		mutable OSharedMutex Mutex;
	};

public:
	explicit OThreadSafeHashMap(uint8 BucketsNum = 13, const Hash& Hasher = Hash())
	    : BucketArray(BucketsNum), MainHasher(Hasher)
	{
		for (auto elem : BucketArray)
		{
			elem = new OBucket;
		}
	}

	OThreadSafeHashMap(const OThreadSafeHashMap& Other) = delete;
	OThreadSafeHashMap& operator=(const OThreadSafeHashMap& Other) = delete;

	TValue GetValueFor(const TKey Key, const TValue& DefaultValue = TValue())
	{
		return GetBucket(Key).GetValueFor(Key, DefaultValue);
	}

	void AddOrUpdateMapping(const TKey& Key, const TValue& Value)
	{
		GetBucket(Key).AddOrUpdate(Value);
	}

	bool Remove(const TKey& Key)
	{
		return GetBucket(Key).Remove(Key);
	}

	OHashMap<TKey, TValue> GetMap() const
	{
		using TUniqueLock = OUniqueLock<OSharedMutex>;
		OVector<TUniqueLock> locks;
		for (auto& bucket : BucketArray)
		{
			locks.push_back(TUniqueLock(bucket->Mutex));
		}
		OHashMap<TKey, TValue> result;
		for (auto& bucket : BucketArray)
		{
			for (auto& data : bucket->Data)
			{
				result.insert(data);
			}
		}
		return result;
	}

private:
	OBucket& GetBucket(const TKey& Key) const
	{
		const auto bucketIdx = MainHasher(Key) % BucketArray.size();
		return *BucketArray[bucketIdx];
	}

	OVector<OUniquePtr<OBucket>> BucketArray;
	Hash MainHasher;
};

} // namespace RenderAPI