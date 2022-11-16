#Low thress passing
class LTPass:
    def __init__(self,xs):
        self.xs=xs
    
    def update(self,xs):
        counter=0
        for x in self.xs:
            self.xs[counter]=.95*x+.05*xs[counter]
            counter+=1
        return self.xs

xs=[1,2]
lt=LTPass(xs)
for x in range(100):
    new=lt.update([3,4])
    print(new)
