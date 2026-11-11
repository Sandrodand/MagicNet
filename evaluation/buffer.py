class Buffer:
    """Delay items by a fixed number of insertions."""
    def __init__(self, n=3):
        """Set the delay to n insertions, using None for the initial empty slots."""
        if n == 0:
            self.buffer = []
        else:
            self.buffer = [None for _ in range(n)]

    def enqueue(self, elem):
        """Insert an item and return the oldest slot, or the same item for a zero delay."""
        if len(self.buffer) == 0:
            return elem
        last = self.buffer[-1]
        self.buffer = [elem] + self.buffer[:-1]
        return last
