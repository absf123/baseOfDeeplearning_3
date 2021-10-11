import unittest
import numpy as np
from Step1.step09 import Variable, square

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0, y1 = f(x0), f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    # 순전파 확인
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y, expected)
    # 역전파 확인
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # 자동 미분으로 test : 기댓값을 몰라도 입력값만 준비하면 기울기 확인 가능
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)  # 수치 미분 결과
        flg = np.allcloas(x.grad, num_grad)  # 수치 미분 결과와 역전파 결과 비교
        self.assertTrue(flg)