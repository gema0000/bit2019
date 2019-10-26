class Person:
    def __init__(self, name, age=10):
        self.name = name
        self.age = age
    def greeting(self):
        print('나는', self.name, '입니다.', self.age, '살 입니다.')

aaa = Person('말똥이', 4)
aaa.greeting()

bbb = Person('존슨')
bbb.greeting()

