import random

import torch
from torch.utils.data import Sampler

from src.data.DataConstants import EOS


class BucketBatchSampler(Sampler):
    def __init__(self, data, batch_size, test_set=False):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.buckets = self.create_buckets(test_set)

    def create_buckets(self, test_set):
        # Create buckets based on length
        length_buckets = {}
        for i, j in enumerate(self.data.indices):
            item = self.data.dataset[j][0]
            # For testing, we only care about the length of the orthography (up to the EOS character)
            if test_set:
                length = torch.nonzero(item == ord(EOS)).item()
            else:
                length = len(item)
            if length not in length_buckets:
                length_buckets[length] = []
            length_buckets[length].append(i)

        # Sort buckets based on size from smallest to largest
        sorted_buckets = dict(sorted(length_buckets.items()))

        # Create batches
        sorted_items = []
        for v in sorted_buckets.values():
            sorted_items.extend(v)

        # debug: dump sorted items to a file
        with open(f"{id(self)}.txt", "w") as file:
            # Convert list to string and write to file
            file.write("\n".join([str(x) for x in sorted_items]))

        batches = []
        for i in range(0, len(sorted_items), self.batch_size):
            batches.append(sorted_items[i:i + self.batch_size])

        # Shuffle all batches
        random.shuffle(batches)
        return batches

    def __iter__(self):
        # Iterate over lists of indices (batches) of dataset's elements
        for batch in self.buckets:
            yield batch

    def __len__(self):
        return len(self.buckets)
