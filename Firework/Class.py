class Fire:
    def __init__(self, x, y, m, type, r, i, c, f, powder):

        self.Pos = (x,y)
        self.mess = m
        self.ExType = type
        self.size = r
        self.Intensity = i
        self.color = c

        self.Fire_force = f
        self.Force = 0
        self.Gunpowder = powder