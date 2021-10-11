# step 역전파 자동화
# Define-by-Run : 딥러닝에서 수행하는 계산들을 계산 시점에 '연결'하는 방식으로 '동적 계산 그래프'라고도 함
import numpy as np
from step06 import Square, Exp

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  # 변수의 창조자, 부모 f

    def set_creator(self, func):
        self.creator = func

    # 역전파 자동화
    def backward(self):
        f = self.creator  # 1. 함수를 가져온다.
        if f is not None:
            x = f.input  # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다.
            x.backward()  # 하나 앞 변수의 backward 메서드를 호출한다. (재귀)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 출력 변수에 창조자를 설정한다. (기억을 시킴) => '연결'을 동적으로 만드는 기법의 핵심
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

    # 계산 그래프의 노드들을 거꾸로 거슬러 올라간다.
    assert y.creator == C  # 여기 step07.py에 Exp, Square class를 따로 정의하지 않으면 step6에서의 Function->Variable로 감 -> creator attirbute가 없음
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # Variable의 인스턴스 벼수 creator에서 바로 앞의 Function으로 건너 간다.
    # 그리고 그 Function의 인스턴스 변수 input에서 다시 하나 더 앞의 Variable로 건너감

    # Define-by-Run : 데이터를 흘려보냄으로써 연결이 규정된다는 뜻
    # 결국 '링크드 리스트'라는 데이터 구조를 이용해 계산 그래프를 표현하고 있는 것

    # 역전파 도전! : 너무 번거로움...
    y.grad = np.array(1.0)
    C = y.creator  # 1. 함수를 가져온다.
    b = C.input  # 2. 함수의 입력을 가져온다.
    b.grad = C.backward(y.grad)  # 3. 함수의 backward 메서드를 호출한다.

    B = b.creator  # 1. 함수를 가져온다.
    a = B.input  # 2. 함수의 입력을 가져온다.
    a.grad = B.backward(b.grad)  # 3. 함수의 backward 메서드를 호출한다.

    A = a.creator  # 1. 함수를 가져온다.
    x = A.input  # 2. 함수의 입력을 가져온다.
    x.grad = A.backward(a.grad)  # 3. 함수의 backward 메서드를 호출한다.
    print(x.grad)

    # 역전파를 자동으로!
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
    print(x.grad)  # y.backward() 메서드를 호출하면 역전파가 자동으로 진행!
