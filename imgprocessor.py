def reverse(self):
    self.modified = True
    self.img = list(map(lambda x: 255 - x, self.img))

def umbral(self, value):
    if self.bw and self.mode == 'L':
        self.modified = True
        self.img = list(map(lambda x: 0 if x < value else 255, self.img))
