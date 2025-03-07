class Buffer:
    def __init__(self, n=3):
        if n == 0:
            self.buffer = []
        else:
            self.buffer = [None for _ in range(n)]

    def enqueue(self, elem):
        if len(self.buffer) == 0:
            return elem
        last = self.buffer[-1]
        self.buffer = [elem] + self.buffer[:-1]
        return last
