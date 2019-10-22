str ='i am a iron man'
f = open('aaa.txt', 'w')
f.write(str)
f.close()

f2 = open('aaa.txt', 'r')
data = f2.read()
print(data)

x = 100
aaa = (lambda x : x**2)
bbb = lambda x  : x**3
print("aaa : ", aaa)
print("aaa(5) : ", aaa(5))
print("bbb(2) : ", bbb(2))

mySimpleFunc = lambda x, y, z : 2*x +3*y + z
print(mySimpleFunc(3,6,9))

mySimpleFunc2 = (lambda a,b,c : 
    a*b+c)
print(mySimpleFunc2(4,3,2))







