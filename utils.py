class Accumulator:
    def __init__(self):
        self.loss = 0
        self.count = 0

    def append(self, loss, batch_size):
        self.loss += loss * batch_size
        self.count += batch_size

    def reset(self):
        self.loss = 0
        self.count = 0
    
    def get(self):
        return self.loss / self.count
