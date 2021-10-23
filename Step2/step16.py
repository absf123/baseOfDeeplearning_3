import numpy as np
from Step1.step09 import as_array

# step16 : 복잡한 계산 그래프, 최근 세대의 함수부터 꺼내도록 구현 -> 아무리 복잡한 계산 그래프라도 올바른 순서로 연전파 이루어짐

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))  # ndarray만 입력받기

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대를 기록 (부모 세대 + 1), f:x -> y (y의 세대는 f의 +1)

    # 미분값 초기화
    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # y.grad = np.array(1.0) 생략

        funcs = []
        seen_set = set()
        def add_func(f):  # 세대 정렬 (중첩 함수 조건 : 1. 감싸는 메서드 안에서만 이용  2. 감싸는 메서드에 정듸된 변수를 사용)
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)  # 큰 세대부터 꺼낼 수 있게 정렬(pop)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 가져온다.
            gys = [output.grad for output in f.outputs]  # output에 담겨있는 미분값들을 list에 담는다
            gxs = f.backward(*gys)  # 역전파 선언, 리스트 언팩
            if not isinstance(gxs, tuple):
                gxs = (gxs,)  # tuple이 아니면 tuple로 변경

            for x, gx in zip(f.inputs, gxs):  # 역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx  # 같은 변수가 아닐 경우
                else:
                    x.grad = x.grad + gx  # 같은 변수일 경우 더해준다

                if x.creator is not None:
                    # funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다
                    add_func(x.creator)
class Function:
    def __call__(self, *inputs):  # 개선1 : 가변 인수 받기
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # 개선2 : unpack
            ys = (ys,)  # 개선2 : 튜플이 아닌 경우 추가 지원
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])  # 입력변수가 여러개일때 가장 큰 generation수 선택
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

    # Add의 역전파는 상류에서 흘러오는 미분값을 그대로 플려보낸다
    def backward(self, gy):
        return gy, gy

def add(x1, x2):
    return Add()(x1, x2)

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


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))  # y = 2x^4
    y.backward()
    print(y.data)  # 32.0
    print(x.grad)  # 64.0