use std::collections::BTreeMap;

/// BTree-backed LRU cache with logn put and get operations
pub struct LRUCache<K, V> {
    max_size: usize,
    timestamp_counter: u64,
    map: BTreeMap<u64, (K, V)>,
    reverse_lookup: BTreeMap<K, u64>,
}

impl<K: Ord + Clone, V> LRUCache<K, V> {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            timestamp_counter: 0,
            map: BTreeMap::new(),
            reverse_lookup: BTreeMap::new(),
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if let Some(&timestamp) = self.reverse_lookup.get(&key) {
            self.map.remove(&timestamp);
        } else if self.map.len() == self.max_size {
            if let Some((&oldest_timestamp, (old_key, _))) = self.map.iter().next() {
                self.reverse_lookup.remove(&old_key);
                self.map.remove(&oldest_timestamp);
            }
        }
    
        self.timestamp_counter += 1;
        self.map.insert(self.timestamp_counter, (key.clone(), value));
        self.reverse_lookup.insert(key, self.timestamp_counter);
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&timestamp) = self.reverse_lookup.get(key) {
            if let Some((_, value)) = self.map.remove(&timestamp) {
                self.timestamp_counter += 1;
                self.map.insert(self.timestamp_counter, (key.clone(), value));
                self.reverse_lookup.insert(key.clone(), self.timestamp_counter);
                return Some(&self.map.get(&self.timestamp_counter).unwrap().1);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_basic_operations() {
        let mut cache = LRUCache::new(3);

        cache.put(1, "one");
        cache.put(2, "two");
        cache.put(3, "three");

        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), Some(&"three"));

        cache.put(4, "four");
        assert_eq!(cache.get(&1), None); // Evicted
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), Some(&"three"));
        assert_eq!(cache.get(&4), Some(&"four"));

        cache.get(&2);
        cache.put(5, "five"); // Evicts 3, as 2 was recently accessed
        assert_eq!(cache.get(&3), None); // Evicted
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&4), Some(&"four"));
        assert_eq!(cache.get(&5), Some(&"five"));
    }

    #[test]
    fn lru_eviction_order() {
        let mut cache = LRUCache::new(2);

        cache.put(1, "one");
        cache.put(2, "two");
        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), Some(&"two"));

        cache.get(&1);
        cache.put(3, "three"); // Evicts 2, as 1 was recently accessed
        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), None); // Evicted
        assert_eq!(cache.get(&3), Some(&"three"));
    }

    #[test]
    fn lru_fuzzy() {
        use rand::Rng;

        let mut cache = LRUCache::new(100);
        let mut rng = rand::thread_rng();
        let mut reference = std::collections::HashMap::new();

        for _ in 0..1000 {
            let operation: u8 = rng.gen_range(0..2); // 0 for put, 1 for get
            let key = rng.gen_range(0..200); // Keys in the range [0, 200)
            let value = rng.gen_range(0..1000); // Random values

            match operation {
                0 => {
                    cache.put(key, value);
                    reference.insert(key, value);

                    if reference.len() > 100 {
                        let evicted_key = reference
                            .keys()
                            .filter(|&&k| !cache.reverse_lookup.contains_key(&k))
                            .cloned()
                            .next();
                        if let Some(evicted_key) = evicted_key {
                            reference.remove(&evicted_key);
                        }
                    }
                }
                1 => {
                    if let Some(&expected_value) = reference.get(&key) {
                        assert_eq!(cache.get(&key), Some(&expected_value));
                    } else {
                        assert_eq!(cache.get(&key), None);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}