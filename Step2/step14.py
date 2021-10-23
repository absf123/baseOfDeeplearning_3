import numpy as np
from Step1.step09 import as_array


# step13의 몇 가지 문제 해결 : 같은 변수 반복 사용으로 인한 문제 해결해야함, 16단계에서 Variable 클래스 완성
# 함수 개선하기 : Variable, Add, Square
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

    # 미분값 초기화
    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # y.grad = np.array(1.0) 생략

        funcs = [self.creator]
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
                    funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다

class Function:
    def __call__(self, *inputs):  # 개선1 : 가변 인수 받기
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):  # 개선2 : unpack
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

    # Add의 역전파는 상류에서 흘러오는 미분값을 그대로 플려보낸다
    def backward(self, gy):
        return gy, gy

def add(x1, x2):
    return Add()(x1, x2)



if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(x, x)
    print('y', y.data)  # 6.0

    y.backward()
    print('x.grad', x.grad)  # 1.0 => 원래는 2.0으로 출력 되어야하는데, 미분값이 덮어 써짐

    x1 = Variable(np.array(3.0))
    y = add(add(x1, x1), x1)
    print('y', y.data)  # 6.0

    y.backward()
    print('x1.grad', x1.grad)  # 1.0 => 원래는 2.0으로 출력 되어야하는데, 미분값이 덮어 써짐

    # 같은 변수를 사용하여 다른 계산 할 경우
    x = Variable(np.array(3.0))
    y = add(x, x)  # 첫 번째 계산
    y.backward()
    print(x.grad)

    x.cleargrad()  # 미분값 초기화
    y = add(add(x, x), x)  # 두 번째 계산
    y.backward()
    print(x.grad)  # 5.0 -> 기존 2.0에 3.0이 더해짐 => 원래는 3.0

