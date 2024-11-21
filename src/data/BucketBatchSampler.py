import random

import torch
from torch.utils.data import Sampler, Subset


class BucketBatchSampler(Sampler):
    def __init__(self, data: Subset, batch_size: int, delimiter=None):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.buckets = self.create_buckets(delimiter)

    def create_buckets(self, delimiter):
        # Create buckets based on length
        length_buckets = {}
        for i, j in enumerate(self.data.indices):
            item = self.data.dataset[j][0]
            # For test sets, we only care about the length of the orthography (i.e. up to the EOS character)
            if delimiter:
                length = torch.nonzero(item == ord(delimiter)).item()
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
