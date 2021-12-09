is_simple_core = True

if is_simple_core:  # 32단계까지는 True
    # __init__.py는 모듈을 import 할때 가장 먼저 실행되는 파일
    from dezero_DJ.core_simple import Variable
    from dezero_DJ.core_simple import Function
    from dezero_DJ.core_simple import using_config
    from dezero_DJ.core_simple import no_grad
    from dezero_DJ.core_simple import as_array
    from dezero_DJ.core_simple import as_variable
    from dezero_DJ.core_simple import setup_variable
else:  # 33단계부터 사용 (일단 채워두고 나중에 변경)
    from dezero_DJ.core_simple import Variable
    from dezero_DJ.core_simple import Function
    from dezero_DJ.core_simple import using_config
    from dezero_DJ.core_simple import no_grad
    from dezero_DJ.core_simple import as_array
    from dezero_DJ.core_simple import as_variable
    from dezero_DJ.core_simple import setup_variable

setup_variable()

# dezero_DJ 패키지를 이용하는 사용자는 반드시 연산자 오버로드가 이루어진 상태에서 Variable을 사용할 수 있다

"""
# from dezero_DJ.core_simple import Variable
from dezero_DJ import Variable # 이렇게 사용할 수 있음
"""