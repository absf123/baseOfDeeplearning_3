# 자연스러운 코드로!
import numpy as np
from Step1.step09 import Variable, as_array


# 가변인수 받도록 수정
class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs  # 출력도 저장한다.
        return outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)  # tuple로 반환하는게 정말 이상함 -> step12 에서 수정 됨

if __name__ == "__main__":
    xs = [Variable(np.array(2)), Variable(np.array(3))]  # list로 준비
    f = Add()
    ys = f(xs)  # ys 튜플
    y = ys[0]
    print(y.data)
    # 너무 부자연스럽고 귀찮음 -> step12에서 개선

