# Variable 구현 -> 변수(상자)
import numpy as np


# 다차원 배열만 취급
class Variable:
    def __init__(self, data):
        self.data = data

"""
3차원 벡터 : 벡터 원소가 3개 ex) [1,2,3]
3차원 배열 : 차원 축이 3개
"""

if __name__ == "__main__":
    data = np.array(1.0)
    x = Variable(data)  # x : 인스턴스, 데이터를 담는 상자
    print(x.data)

    x.data = np.array(2.0)
    print(x.data)

    # numpy 다차원 배열(텐서)
    x = np.array(1)
    print(x.ndim)  # 'number of dimensions' : 다차원 배열의 차원 수

    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x.ndim)