# step8 재귀에서 반복문으로
import numpy as np

# 기존 Variable에서 재귀 -> 반복문으로 구현

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 출력값의 creator를 가져옴 -> 입력값의 creator로 바꿔줌
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 함수를 가져온다.
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)  # backward 메서드를 호출한다.

            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 출력 변수에 창조자를 설정한다. (기억을 시킴) => '연결'을 동적으로 만드는 기법의 핵심, func가 찍힘
        self.input = input
        self.output = output  # 출력도 저장한다.
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 역전파
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)  # 재귀는 함수를 재귀적으로 호출할 때마다 중간 결과를 메모리에 유지하면서 처리를 이어감 -> 반복문 방식의 효율이 더 좋은 이유
