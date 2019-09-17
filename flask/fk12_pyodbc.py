# 모듈을 불러옵니다.
import pyodbc as pyo

# 연결 문자열을 세팅합니다.
server = 'localhost'
database = 'bitdb'
username ='bit'
password ='1234'

# 데이터 베이스를 연결합니다.
cnxn = pyo.connect('DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' +server+ 
                      '; PORT=1433; DATABASE=' +database+ 
                      ';UID=' +username+ 
                      ';PWD=' +password
                     )
# 커서를 만듭니다.
cursor = cnxn.cursor()

# 커서에 쿼리를 입력해 실행시킵니다.
tsql = "SELECT * FROM iris2;"
with cursor.execute(tsql):
    # 한행을 가져옵니다.
    row = cursor.fetchone()
    # 행이 존재할 떄까지, 하나씩 행을 증가시키면서 모든 컬럼을 공백으로구분해 출력합니다.
    while row:
        print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
              str(row[3]) + " " + str(row[4])) 

cnxn.close()







