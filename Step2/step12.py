import numpy as np
from Step1.step09 import Variable, as_array

"""
1. '사용하는 사람'을 위한 개선
2. '구현하는 사람'을 위한 개선
"""

# 입력, 출력을 리스트나 튜플을 안 거치도록 변환
class Function:
    def __call__(self, *inputs):  # 개선1 : 가변 인수 받기
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 개선2 : unpack
        if not isinstance(ys, tuple):
            ys = (ys,)  # 개선2 : 튜플이 아닌 경우 추가 지원
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs  # 출력도 저장한다.
        return outputs if len(outputs) > 1 else outputs[0]  # 개선1 : 리스트의 원소가 하나라면 첫 번재 원소를 반환한다.

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

class Divide(Function):
    def forward(self, x0, x1):
        return x0 / x1

class Multiply(Function):
    def forward(self, x0, x1):
        return x0 * x1


# add 함수로 구현
def add(x0, x1):
    return Add()(x0, x1)

def divide(x0, x1):
    return Divide()(x0, x1)

def multiply(x0, x1):
    return Multiply()(x0, x1)


if __name__ == "__main__":

    # 개선 1
    x0, x1 = Variable(np.array(2)), Variable(np.array(3))
    f = Add()
    y = f(x0, x1)  # 가변 인수 사용가능
    y.backward()
    print(y.data)

    # 개선2
    x0, x1 = Variable(np.array(2)), Variable(np.array(3))
    y = add(x0, x1)  # Add 클래스 생성 과정이 감춰짐
    print(y.data)