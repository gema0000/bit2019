class Bicycle():
    def move(self, speed):
        print("자전거 : 시속 {0}킬로미터로 전진".format(speed))
    def turn(self, direction):
        print("자전거 : {0}회전 ".format(direction))    
    def stop(self):
        print("자전거({0}, {1}): 정지 ".format(self.wheel_size, self.color))

my_bicycle = Bicycle()
my_bicycle.wheel_size =26
my_bicycle.color = 'black'

print("바퀴 크기 : ", my_bicycle.wheel_size)




