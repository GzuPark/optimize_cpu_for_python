# Python을 사용하는 ML/AI 개발자를 위한 CPU 최적화

NAVER DEVIEW 2023 중 [ML/AI 개발자를 위한 단계별 Python 최적화 가이드라인](https://deview.kr/2023/sessions/541) 세션의 코드를 직접 실행하여 가능성을 확인해보는 목적입니다. 대부분의 코드는 세션에 나오는 내용이며, CPU 최적화를 위한 C++ 코드는 세션에서 설명한 내용을 담기 위해 구현에 집중했습니다. 이 전에도 C/C++로 작성하여 Python 또는 Dart에서 호출할 때 유니버셜한 라이브러리 사용 목적과 더불어 더 빠른 런타임을 얻기 위한 노력을 해왔고, 특히 위에서 언급한 세션에서는 OpenCV와 NumPy를 사용하여 라이브러리로 호출하는 가이드를 제시해줌으로써 딥러닝 모델 학습의 데이터로더 CPU 병목을 줄이기 위한 용도 또는 모델 추론 과정중 전처리 속도 개선 등에 적용하기 좋다고 생각합니다.

## 설정
제가 실험한 환경은 M1 맥북프로이며, Python은 3.11, 컴파일을 위한 GCC는 12.2.0를 사용했습니다. 사용하는 Python 패키지는 아래 명령어로 설치하면 됩니다.

```shell
python -m pip install -r requirements.txt
```

### MacOS
컴파일을 위해 MacOS [Homebrew](https://brew.sh/)를 사용하여 `gcc`, `opencv`, `numpy`를 설치해야 합니다.

```shell
brew install gcc opencv numpy
```

### PyCharm
PyCharm에서 `line-profiler`를 사용하기 위해서는 plugin에서 별도의 [Line Profiler](https://plugins.jetbrains.com/plugin/16536-line-profiler)를 설치해야 합니다.

## 사용 방법
환경 설정이 됐다면, 아래 명령어로 필요한 모듈을 컴파일 해주세요. 그러면, 현재 경로에 OS와 Python 버전에 맞는 컴파일드된 파일을 확인할 수 있습니다.

```shell
python setup.py build_ext --inplace
```

### 권장 사용 파일
- `extract_color.py`: 이 파일은 line profiler를 사용할 수 있는 파일입니다. IDE에 맞게 설정해주세요.
- `image_crop.py`: OCR을 위한 이미지, 박스를 가지고 작업을 합니다. OpenCV 작업을 C++로 구현할 경우 Python보다 더 빠르게 동작하는 경우가 있다는 것을 보여줍니다. 
- `tutorial.ipynb`: _**(권장)**_ 각 파일의 주요 함수를 불러와 속도를 비교하는 파일입니다.

```shell
jupyter-lab tutorial.ipynb
```

## 결과

### Line profiler with `extract_color.py`
```text
Naive runs 100 times: 1.184 second
Fast runs 100 times: 0.912 second
Fater runs 100 times: 0.721 second
Cython runs 100 times: 0.963 second
```

### Optimize Opencv with `image_crop.py`
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
