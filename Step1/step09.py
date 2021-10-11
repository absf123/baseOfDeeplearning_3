import numpy as np

# 함수 개선하기 : Variable, Function
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))  # ndarray만 입력받기

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # y.grad = np.array(1.0) 생략

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
        output = Variable(as_array(y))  # ndarray()로 변환
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


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# ndarray()만 취급하기
def as_array(x):
    if np.isscalar(x):
        return np.array(x)  # float나 int같은 scalar면 array로 바꾸자
    return x




if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))  # 연속해서 사용 가능, class마다
    # y.grad = np.array(1.0)  # 이제 생략해도 사용 가능
    y.backward()
    print(x.grad)

    # numpy가 의도한 결과 : 0 차원 ndarray 인스턴스 사용하면 결과의 data type이 float64 or float32
    x = np.array(1.0)  # 0차원 ndarray
    y = x ** 2
    print(type(x), x.ndim)
    print(type(y))
    """
    <class 'numpy.ndarray'> 0
    <class 'numpy.float64'>
    """

