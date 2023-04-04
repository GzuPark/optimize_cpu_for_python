# ML/AI 개발자를 위한 Python CPU 최적화 가이드
NAVER DEVIEW 2023의 [ML/AI 개발자를 위한 단계별 Python 최적화 가이드라인](https://deview.kr/2023/sessions/541) 세션의 코드를 실행하여 최적화 가능성을 확인합니다.
주로 세션에서 소개된 내용을 담고 있으며, CPU 최적화를 위해 C++ 코드를 구현하였습니다.
이를 통해 Python에서 호출할 때 유니버설한 라이브러리를 사용하고, 더 빠른 런타임을 얻기 위해 노력했습니다.
특히, 이 세션에서는 OpenCV와 NumPy를 사용하여 라이브러리를 호출하는 방법을 소개함으로써, 딥러닝 모델 학습의 데이터로더 CPU 병목을 줄이고 모델 추론 과정에서 전처리 속도를 개선하는데 도움이 됩니다.

## 환경 설정
실험 환경은 M1 맥북프로이며, Python 3.11 및 GCC 12.2.0을 사용했습니다. 필요한 Python 패키지는 다음 명령어로 설치할 수 있습니다.

```shell
python -m pip install -r requirements.txt
```

### MacOS
컴파일을 위해 MacOS [Homebrew](https://brew.sh/)를 사용하여 `gcc`, `opencv`, `numpy`를 설치해야 합니다.

```shell
brew install gcc opencv numpy
```

### PyCharm
PyCharm에서 `line-profiler`를 사용하기 위해서는 plugin에서 [Line Profiler](https://plugins.jetbrains.com/plugin/16536-line-profiler)를 설치해야 합니다.

## 실행 방법
환경 설정이 완료되면, 다음 명령어로 필요한 모듈을 컴파일할 수 있습니다. 그럼 현재 경로에 OS와 Python 버전에 맞는 컴파일된 파일을 확인할 수 있습니다.

```shell
python setup.py build_ext --inplace
```

### 권장 사용 파일
- `extract_color.py`: line profiler를 사용할 수 있는 파일입니다. IDE에 맞게 설정해주세요.
- `image_crop.py`: OCR을 위한 이미지, 박스를 가지고 작업을 합니다. OpenCV 작업을 C++로 구현할 경우 Python보다 더 빠르게 동작하는 경우가 있다는 것을 보여줍니다. 
- `tutorial.ipynb`: _**(권장)**_ 각 파일의 주요 함수를 불러와 속도를 비교하는 파일입니다.

```shell
jupyter-lab tutorial.ipynb
```

## 결과

### `extract_color.py`
```text
Naive runs 100 times: 1.184 second
Fast runs 100 times: 0.912 second
Fater runs 100 times: 0.721 second
Cython runs 100 times: 0.963 second
```

### `image_crop.py`
```text
Python runs 100 times: 3.706 second
C++ wrapper runs 100 times: 1.748 second
OpenMP runs 100 times: 0.604 second
```

## 참고
- NAVER DEVIEW 2023: [ML/AI 개발자를 위한 단계별 Python 최적화 가이드라인](https://deview.kr/2023/sessions/541)
- [Extending Python with C or C++](https://docs.python.org/3/extending/extending.html), _Official Python document_
- [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/index.html), _Official NumPy document_
- [OpenMP 지시문](https://learn.microsoft.com/ko-kr/cpp/parallel/openmp/reference/openmp-directives?view=msvc-170), _Microsoft Document_
- 이미지 소스: [만들어쓰는 개발진스](https://devjeans.dev-hee.com/)
