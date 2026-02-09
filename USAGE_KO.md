<div align="center">

<img src="assets/joker.png" alt="Joker Llamas" width="300"/>

<p align="center">
  <a href="http://114.110.134.81:7860/" target="_blank">데모 페이지</a>
</p>

</div>

***

# LLaMA-3 Joker 한국어 사용 가이드

## 소개

LLaMA-3 Joker는 생성형 AI 모델이 한국어 환경에서 외부의 비윤리적 공격에 얼마나 효과적으로 대응하는지 테스트하기 위해 설계된 모델입니다.

그동안 LLM 모델의 안전성과 윤리적 행동을 보장하기 위한 노력이 꾸준히 이루어져 왔습니다. 연구자들은 주로 사람이 직접 모델에 부적절한 입력을 넣고 응답을 평가하는 방식에 의존해 왔지만, 이는 시간이 많이 소요되고 정신적으로도 매우 피로한 작업입니다. Joker는 이러한 인간의 공격을 AI로 대체하여 모델 응답 평가를 자동화하는 것을 목표로 합니다.

## 모델 정보

Meta의 LLaMA-3 (8B) 체크포인트를 AI Hub의 ['텍스트 윤리검증 데이터'](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=558) 데이터셋으로 파인튜닝한 모델입니다. 이 데이터셋에는 다양한 유형의 혐오 발언 사례가 포함되어 있어, 포괄적이고 견고한 테스트가 가능합니다.

## 빠른 시작

### 1. 환경 준비

PyTorch와 CUDA가 설치된 conda 환경에서 레포지토리를 클론합니다.

```bash
git clone https://github.com/tunib-ai/joker.git
cd joker
```

### 2. 의존성 설치

```bash
pip install -e .
```

필요한 패키지:
- `torch` — PyTorch
- `transformers` — Hugging Face Transformers
- `fire` — Python Fire (CLI 인터페이스)

### 3. 모델 다운로드

[Hugging Face 모델 허브](https://huggingface.co/tunib/llama-3-joker)에 방문하여 모델 접근 권한을 신청합니다.

> **참고:** 악용 방지를 위해 Joker 체크포인트는 개인이 아닌 **조직 단위**로만 접근이 가능합니다. Hugging Face의 모델 접근 요청 시스템을 통해 신청하면, 내부 검토 후 승인 시 접근 권한이 부여됩니다.

### 4. 실행

```bash
python3 example.py
```

실행하면 대화형 인터페이스가 시작됩니다. `exit`를 입력하면 종료됩니다.

## 코드 사용법

### 기본 사용

```python
from example import Joker

# 모델 로드
joker = Joker("tunib/llama-3-joker")

# 대화 메시지 구성
messages = [
    {"role": "user", "content": "안녕하세요"},
]

# 응답 생성
response = joker(messages=messages, category="ABUSE")
print(response)
```

### 카테고리

모델은 다음 7가지 카테고리를 지원합니다:

| 카테고리 | 설명 |
|----------|------|
| `ABUSE` | 욕설 |
| `CENSURE` | 비난 |
| `CRIME` | 범죄 관련 |
| `DISCRIMINATION` | 차별 |
| `HATE` | 혐오 |
| `SEXUAL` | 성적 표현 |
| `VIOLENCE` | 폭력 |

카테고리를 지정하지 않으면 위 목록에서 무작위로 선택됩니다.

### 디코딩 옵션 커스터마이징

```python
decoding_options = {
    "num_beams": 3,
    "max_new_tokens": 128,
    "temperature": 1.0,
    "top_p": 0.8,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "early_stopping": True,
}

response = joker(
    messages=messages,
    category="HATE",
    decoding_options=decoding_options,
)
```

### CLI 옵션

```bash
# 기본 실행
python3 example.py

# 특정 카테고리 지정
python3 example.py --category ABUSE

# 다른 모델 경로 사용
python3 example.py --pretrained_model_name_or_path "모델/경로"
```

## 데모

[데모 페이지](http://114.110.134.81:7860/)에서 모델을 직접 체험할 수 있습니다.

<img src="assets/demo.png" height="300">

## 주의사항

> **경고:** 이 모델을 상식과 사회적 규범에 어긋나는 목적으로 사용하는 것은 금지되며, 그로 인한 모든 책임은 사용자에게 있습니다.

## 인용

이 라이브러리를 프로젝트나 연구에 활용할 경우, 아래와 같이 인용해 주세요:

```bibtex
@misc{llama-3-joker
  author       = {Kim, Soohwan and Park, Kyubyong},
  title        = {LLaMA-3 Joker},
  howpublished = {\url{https://github.com/tunib-ai/joker}},
  year         = {2024}
}
```

## 문의

문의사항이나 이슈가 있으시면 tunibridge@tunib.ai 로 연락해 주세요.

## 감사의 말

Joker 학습을 위한 GPU는 [Common Computer](https://comcom.ai/)에서 제공해 주셨습니다.
