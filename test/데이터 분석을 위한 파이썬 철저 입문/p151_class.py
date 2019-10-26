robot_name = 'R1'
robot_pos = 0

def robot_move():
    global robot_pos
    robot_pos = robot_pos + 1
    print("{0} position: {1}".format(robot_name, robot_pos))

robot_move()
print("==================")

class Robot():
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
    def move(self):
        self.pos = self.pos+1
        print("{0} position : {1}".format(self.name, self.pos))

robot1 = Robot('R1', 0)
robot2 = Robot('R2', 10) 
robot1.move()
robot2.move()
       








