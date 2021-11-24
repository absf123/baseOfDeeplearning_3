# 메모리를 절약해보자! : 1. 역전파 시 사용하는 메모리양 줄이기 2. 역전파가 필요ㅎ없는 경우용 모드 제공

# 1. 필요없는 미분값 삭제 : 현재 DeZero에서는 모든 변수가 미분값을 변수에 저장
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

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    # Add의 역전파는 상류에서 흘러오는 미분값을 그대로 플려보낸다
    def backward(self, gy):
        return gy, gy

def add(x1, x2):
    return Add()(x1, x2)


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)  # None, None
print(x0.grad, x1.grad)  # 2.0, 1.0
# 우리가 관심있는 것 x0, x1의 미분값, y, t의 미분값은 필요하지 않음


# 2. with 문을 활용한 모드 전환
import contextlib

@contextlib.contextmanager  # 문맥을 판단하는 함수가 만들어짐
def config_test():
    print('start')  # 전처리 : with 문 안으로 들어갈때
    try:
        yield  # 예외처리 필요
    finally:
        print('done')  # 후처리 : with 문 빠져나올때

with config_test():
    print('process...')

# 위를 참고해서 using_config

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)  # gettar로 Config 클래스에서 꺼내오기
    setattr(Config, name, value)  # setattr로 새로운 값을 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value)

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


with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
    # with 안에서만 '역전파 비활성 모드
# with 블록을 벗어나면 '역전파 활성 모드'  => with torch.no_grad()와 동일한 역할임

# 위에 처럼 매번 길게 작성하기 귀찮으니 짧게 처리하자

def no_grad():
    return using_config('enable_backprop', False)

# with torch.no_grad()와 동일 : 이제 기울기가 필요 없을 때는 no_grad 함수를 호출하면 됨 (모드 전환)
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)



