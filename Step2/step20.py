# 연산자 오버로드 추가 2021.12.9
import numpy as np
import weakref
from Step1.step09 import as_array

class Config:
    enable_backprop = True  # True : 역전파 모드, False라면 중간 계산 결과는 사용 후 곧바로 삭제(정확하게는 다른 객체에서의 참조가 없어지는 시점에 메모리에서 삭제)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # enable_backprop이 True일때 역전파 실행
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)  # 계산들의 '연결'을 만드는데, 마찬가지로 '역전파 비활성 모드'에서는 필요 없습니다.
            self.inputs = inputs  # 결괏값을 보관하는 로직 -> 역전파 계산시 사용 : 때로는 미분값이 필요 없는 경우도 있음 ex) inference => Config에서 enable_backprop 속성으로 조절
            # self.outputs = outputs  # 출력도 저장한다.
            self.outputs = [weakref.ref(output) for output in outputs]  # self.outputs가 대상을 약한 참조로 가리키게 변경 -> 다른 클래스에서 Function 클래스의 outputs를 참조하는 코드도 수정해야함
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    # gy = dL/dy  (d가 아니라 편미분임)
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

# 함수로 사용
def mul(x0, x1):
    return Mul()(x0, x1)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    # Add의 역전파는 상류에서 흘러오는 미분값을 그대로 플려보낸다
    def backward(self, gy):
        return gy, gy

def add(x1, x2):
    return Add()(x1, x2)

# a = Variable(np.array(3.0))
# b = Variable(np.array(2.0))
# c = Variable(np.array(1.0))
#
# y = add(mul(a, b), c)  # add와 mul을 같이 사용할 수 있게됨 -> 이걸 연산자로 표현해보자 : Variabel의 확장
# y.backward()
#
# print(y)
# print(a.grad)
# print(b.grad)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name  # 변수에 이름 붙여주기 => 이후 계산 그래프 시각화할때 변수 이름을 그래프에 표시할 수 있음 (26,27장)
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # 미분값 초기화
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):  # 중간 변수에 대해서는 미분값을 제거하는 모드 추가
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]  # 수정전
            gys = [output().grad for output in f.outputs]  # output이 약한 참조니 참조 데이터로 접근하려면 ()을 붙여야 함
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조 (weakref) : output 생각해보자

    # property (데코레이터) : shape 메서드를 인스턴스 변수처럼 사용할 수 있음 => x.shape()을 x.shape으로 사용가능케함, 필요한건 더 추가해보기
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype  # dtype 지정하지 않으면 ndarray는 float64,int64로 초기화 됨, 신경망에서는 float32를 사용하는 경우가 많음

    def __len__(self):
        return len(self.data)  # 특수메서드 사용해 Variable 인스턴스도 len함수 사용할수 있게함

    # Variable의 내용 쉽게 확인할 수 있게, 데이터 내용 출력
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    # 연산자 오버로드
    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)


if __name__ == "__main__":
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    # Variable.__mul__ = mul  # 파이썬에서는 함수도 객체이므로 이와 같이 함수 자체를 할당할 수 있다
    # Variable.__add__ = add

    y = a * b + c  # Variable이랑 Function 클래스는 둘 중 하나라도 변경되면 매 step마다 새로 생성... (참조되는 부분에서 계속 꼬임)
    y.backward()
    print(y)
    print(a.grad)
    print(b.grad)

    """
    variable(7.0)
    2.0
    3.0
    """