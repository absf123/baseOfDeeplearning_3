# step17

class obj:
    pass

def f(x):
    print(x)

a = obj()  # 변수의 대입 : 참조 카운트 1
f(a)  # 함수에 전달 : 함수 안에서는 참조 카운트 2
# 함수 완료 : 빠져나오면 참조 카운트 1
a = None  # 대입 해제 : 참조 카운트 0 -> 0이 되면 해당 객체는 즉시 메모리에서 삭제

# 수많은 메모리 문제 해결 가능
a = obj()  # 각 객체 참고카운트 1
b = obj()
c = obj()

a.b = b  # a가 b를 참조
b.c = c  # b가 c를 참조

a = b = c = None  # a -> 0, b,c -> 1 : 이때 a가 사라지면서 b의 참조 카운트가 1에서 0으로 줄고 c도 0이 되어 전부 도미노처럼 삭제됨
# 위와 같은 방식이 파이썬의 참조 카운트 방식 메모리 관리
# 하지만 순환 참조는 해결 할 수 없음

## 순환 참조
a = obj()
b = obj()
c = obj()

# a가 b를 참조하고 b는 c를 참조하고 c는 a를 참조함
a.b = b
b.c = c
c.a = a

a = b = c = None  # 순환참조시에는 이때, a,b,c 참조 카운터가 1이 됨 : 하지만 None이기 때문에 어느 것에도 접근할 수 없고 불필요한 객체

""" 그럼 어떻게 해야할까? """ # Garbage collector의 등장
# 사실 GC는 참조 카운트와 달리 메모리가 부족해지는 시점에 파이썬 인터프리터에 의해 자동으로 호출 -> 일반적으로 순환 참조를 의식할 필요가 특별히 없음
# 하지만 ML, DL에서는 메모리가 중요한 자원으로 최대한 순환참조를 만들지 않는게 좋음

# 현재 DeZero에 존재하는 순환참조 : Function 인스턴스는 Variable 인스턴스 입력과 출력을 참조하고 Variable은 Function의 인스턴스를 참조하여 순환참조가 생김
# weakref으로 해결하자
import weakref
import numpy as np
from Step1.step09 import as_array

a = np.array([1, 2, 3])
b = weakref.ref(a)  # 다른 객체를 참조하되 참조 카운트는 증가시키지 않음
print(b)  # 값이 아니라 ndarray를 가리키는 약한 참조임을 확인만 가능
print(b())  # 참조된 데이터에 접근하려면 b() 사용

a = None
print(b)  # dead라고 출력 됨

# Function에 추가
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        # self.outputs = outputs  # 출력도 저장한다.
        self.outputs = [weakref.ref(output) for output in outputs]  # self.outputs가 대상을 약한 참조로 가리키게 변경 -> 다른 클래스에서 Function 클래스의 outputs를 참조하는 코드도 수정해야함
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # 미분값 초기화
    def cleargrad(self):
        self.grad = None

    def backward(self):
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

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data  # Function에서 가변 인수를 받는거에 대한 수정
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

# 순환참조가 없어진 코드 실행
for i in range(10):
    x = Variable(np.random.randn(10000))  # huge data
    y = square(square(square(x)))  # 복잡한 계산을 수행