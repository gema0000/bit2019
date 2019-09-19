data = enumerate((1,1,2,4))
print(data)
print(data,type(data))

for i, value in data:
    print(i, ":", value)

data = enumerate({1,2,3})
for i, value in data:
    print(i, ":", value)


data = enumerate([1, 2, 3])
for i, value in data:
    print(i, ":", value)
print()

dict1 = {'이름': '한사람', '나이': 33}
data = enumerate(dict1)
for i, key in data:
    print(i, ":", key, dict1[key])
print()

data = enumerate("재미있는 파이썬")
for i, value in data:
    print(i, ":", value)
print()

aaa = [1,2,3,4,7,91]

for i, value in enumerate(aaa):
    print(i, ' : ', value)

bbb = ['dnririr', '갸갸갹', 'dfkfg', '멀바', '을ff']
for i, value in enumerate(bbb):
    print(i, value)
    
        