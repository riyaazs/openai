import numpy

class SumTree:
    write = 0

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 , dtype=float)
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update_one(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update_one(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def update(self, indices, ps):
        for idx, p in zip(indices, ps):
            self.update_one(idx, p)

    def get(self, s):
        idx = [self._retrieve(0, i) for i in s]
        dataIdx = [i - self.capacity + 1 for i in idx]

        return (idx, self.tree[idx], self.data[dataIdx])

    def sample(self, n):
        return self.get(numpy.random.random(n)*self.total())
