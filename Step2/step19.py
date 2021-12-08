import numpy as np

# step 19 : Dezero Variable 클래스를 numpy의 ndarray처럼!

# 변수에 이름 정해주기
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

if __name__ == "__main__":
    x = Variable(np.array([1,2,3]))
    print(x)  # variable([1 2 3])