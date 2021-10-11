# function 구현
import numpy as np
from step01 import Variable


# Variable 인스턴스를 변수로 다룰 수 있는 함수
class custom_Function:
    def __call__(self, input):
        x = input.data  # 데이터 꺼내기
        y = x ** 2  # 실제 계산
        output = Variable(y)
        return output

# general한 Function 구현
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()  # '이 메서드는 상속하여 구현해야한다'

# Function 상속받고, forward 구현
class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = custom_Function()
    y = f(x)  # __call__ 호출

    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
