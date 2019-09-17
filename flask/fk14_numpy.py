# 모듈을 불러옵니다.
import pymssql as ms
import numpy as np

# 데이터베이스에 연결합니다.
conn = ms.connect(server='127.0.0.1', user='bit', password='1234', database='bitdb')

# 커서를 만듭니다.
cursor = conn.cursor()

# 커서에 쿼리를 입력해 실행 시킵니다.
cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
print(row)
conn.close()

aaa = np.asarray(row)
print(aaa)
print(aaa.shape)
print(type(aaa))

np.save('test_aaa.npy', aaa)

'''
# 한행을 가저옵니다.
row = cursor.fetchone()
# print(type(row))        # tuple

# 행이 존재할 때까지, 하나씩 행을 증가시키면서 1번째 컬럼을 숫자 2째번 컬럼을 문자로 출력합니다.
while row:
    print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    # print(row)
    row = cursor.fetchone()

# 연결을 닫습니다.
conn.close()
'''