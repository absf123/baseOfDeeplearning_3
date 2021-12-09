if '__file__' in globals():  #  __file__ 이라는 전역 변수가 정의되어 있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 현재 파일이 위치한 디렉터리의 부모 디렉터리를 모듈 검색 경로에 추가
    # -> 이로써 파이썬 명령어를 어디에서 실행하든 dezero_DJ 디렉토리의 파일들은 제대로 import할 수 있게 됨
    # 이건 현재 개발 중인 dezero_DJ 디렉토리를 import 하기위해 임시로 사용
    # Dezero가 패키지로 설치된 경우라면 DeZero패키지가 파이썬 검색 경로에 추가 => 그러면 이렇게 수동으로 할 필요없음
    # 다만 colab같은 환경때문에 사용중


import numpy as np
from dezero_DJ import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()
print(y)
print(x.grad)

"""
variable(16.0)
8.0
"""