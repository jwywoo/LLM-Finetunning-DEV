# 5/3/1 Exercise Routine Generation 

## Contributor

<div align="center">

| ![Wooyong Jeong](https://github.com/jwywoo.png?size=300) |
|:--------------------------------------------------------:|
|           [GitHub](https://github.com/jwywoo)            |
|     [jwywoo26@egmail.com](mailto:jwywoo26@mail.com)      |
|                  **Wooyong Jeong: Woo**                  |

</div>

## 프로젝트 소개

### 저장소

```shell
data/ : 프로젝트에서 사용된 데이터들이 닮겨있습니다.
├── dataset
└── src
fine-tuning-src/ : 파인튜닝 관련 코드들을 확인할 수 있습니다. 
└── task1
generator-api/ : 파인튜닝된 모델을 사용할 수 있는 api로 전환하여 만들었습니다. *모델의 경우 HuggingFace 혹은 Google Drive에서 가져와서 사용하여야합니다.
├── src
└── requirements.txt
.gitignore
README.md
```

### 개발환경 및 기술스택

#### 개발환경

- Fine-tuning & Data collection
  - GPU: Colab A100
  - 사용모델: EleutherAI/polyglot-ko-1.3b
  - Fine-tuning 방식: HuggingFace PEFT-LoRa & Trainer
- API
  - FastAPI

#### 사용기술스텍

<div>
  <img alt="PYTHON" src="https://img.shields.io/badge/python-3776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white" alt="GIT">
  <img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" alt="VS Code">
  <img src="https://img.shields.io/badge/Google%20colab-F9AB00?style=for-the-badge&logo=Google%20colab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  <img alt="FASTAPI" src="https://img.shields.io/badge/fastapi-009688.svg?&style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img alt="HUGGINGFACE" src="https://img.shields.io/badge/huggingface-FFD21E.svg?&style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img alt="HUGGINGFACE" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/>
</div>

### 사용법

1. `requirements.txt`를 설치 -> gpu환경과 cpu환경을 대비했으나 gpu환경일 경우 `pip install bitsandbytes` 해주셔야합니다.
2. `generator-api/src/model/task1` 디렉토리를 만들고 공유된 `lora-output-v2`를 옮겨야 합니다.
3. 터미널에서 `src/` 안에서 `uvicorn main:app --loop uvloop --reload`로 FastAPI를 실행하시고 `localhost:8000/docs`에서 api 테스트를 진행할 수 있습니다.

Task1과 Task2로 나누어 Fine-tuning을 진행했습니다. 완료된 Task1를 `gen/routine/rate`을 통해 사용할 수 있습니다.

![test](/static/localhost_doc_api.png)

예시 입력

```shell
# Example 1
{
  "experience": 1,
  "gender": "남성",
  "purpose": 1,
  "rm_weight": {
    "bench_press": 40,
    "squat": 50,
    "dead_lift": 55,
    "over_head_press": 30
  }
}

# Example 2
{
  "experience": 2,
  "gender": "여성",
  "purpose": 3,
  "rm_weight": {
    "bench_press": 60,
    "squat": 70,
    "dead_lift": 75,
    "over_head_press": 40
  }
}

# Example 3
{
  "experience": 3,
  "gender": "남성",
  "purpose": 2,
  "rm_weight": {
    "bench_press": 80,
    "squat": 100,
    "dead_lift": 105,
    "over_head_press": 45
  }
}

# Example 4
{
  "experience": 4,
  "gender": "여성",
  "purpose": 4,
  "rm_weight": {
    "bench_press": 90,
    "squat": 110,
    "dead_lift": 120,
    "over_head_press": 50
  }
}

# Example 5
{
  "experience": 5,
  "gender": "남성",
  "purpose": 6,
  "rm_weight": {
    "bench_press": 130,
    "squat": 160,
    "dead_lift": 170,
    "over_head_press": 60
  }
}
```

## 개발과정

### 분석

5/3/1 프로그램은 아래와 같은 특징을 가지고 있습니다.

분석과정을 통해 아래와 같은 결정을 했습니다.

1. **챗봇의 형태가 아닌 정형적인 입력을 기반으로 생성하는 API형태로 만든다.**
    - App을 사용해본결과 회원가입과정에서의 입력과 추가적인 입력만 있다면 5/3/1 프로그램 루틴 생성에 있어 충분하다고 판단하였습니다.

2. **루틴 생성 모델에 역할을 나누어 두개의 모델을 만든다.**
    - 5/3/1 프로그램의 경우 증량계획과 타격부위와 보조운동 선정을 기반으로 Task를 나눌수 있다고 판단했습니다.
      - Task1: 1RM(지원자의 경우 TM을 선호), 운동경험, 운동목적을 기준으로 한달간 증량계획 생성
      - Task2: 생성된 증량계획과 운동목적, 한주간 운동횟수와 같은 정보들을 기반으로한 구체적인 1달 운동 루틴 생성

위와 같이 Task를 분활한 이유는 LLM Application을 개발해 오면서 하나의 모델 혹은 요청에 다양한 내용이 닮겨 있을 경우 생성결과가 안좋아지는 경험이 다수 있었습니다.
또한 최근 Multi-Agent와 MCP 같은 기술들이 트렌드로 자리잡고 있기 때문에 관련된 내용을 바탕으로 확장성 또한 챙길수 있다고 생각했습니다.
마지막으로 Task1에서 사용되는 정보와 Task2에서 사용되고 생성되는 정보가 차이가 있고 증량계획을 바탕으로 Task2의 생성과정에 있어 도움이 될수 있다고 판단했습니다.

### 데이터 수집

데이터의 경우 아래의 링크의 내용을 바탕으로 수집하지 않고 생성하였습니다.

그리고 아래와 같은 Task1 학습데이터를 만들었습니다.

```shell
## 질문
-5/3/1 프로그램
- **운동경험**: 1
- **성별**: 남성
- **운동목표**: 4
- **1RM**:
  - 벤치프레스: 30kg
  - 스쿼트: 30kg
  - 데드리프트: 30kg
  - 오버헤드프레스: 30kg

## 답변
{
  "program": "5/3/1 프로그램",
  "init_weight_rate": "45",
  "increase_rate_week": "10",
  "increase_rate_set": "5",
  "deloading_rate": "50",
  "weekly_weight_increase_plan": "Week1\nSET1 45%x5\nSET2 50%x5\nSET3 55%x5+\n\nWeek2\nSET1 55%x3\nSET2 60%x3\nSET3 65%x3+\n\nWeek3\nSET1 65%x5\nSET2 70%x3\nSET3 75%x1+\n\nWeek4\nSET1 50%x5\nSET2 55%x5\nSET3 60%x5\n"
}
<END>
```

### Fine-tuning 과정

가용할 수 있는 컴퓨팅 리소스의 한계로 Fine-tuning을 다양하게 시도해 보지는 못했습니다. 그래서 가장 기본적인 LoRa Config 구조를 바탕으로 진행했습니다.

LoRa 방식 선정이유: LoRa 방식을 선정한 이유의 경우 1주일이라는 기간에 2개의 모델 Fine-tuning하기 위해서는 효율적이고 신속한 방식을 사용해야했습니다.

Polyglot-Ko-1.3B 선정이유: 일전의 Flux Fine-tuning의 경우 사용할 수 있는 ai-toolkit이 존재했습니다. 반면에 LLM의 경우 PEFT를 통해 진행해야 했다보니 사용횟수와 자료가 가장 많은 Model을 선정했습니다.

참고자료
- [Zero-Shot Learning vs. Few-Shot Learning vs. Fine-Tuning: A technical walkthrough using OpenAI's APIs & models](https://labelbox.com/guides/zero-shot-learning-few-shot-learning-fine-tuning/)
- [LLM의 다양한 SFT 기법: Full Fine-Tuning, PEFT (LoRA, QLoRA)](https://ariz1623.tistory.com/348)

## 결과 및 회고

안타깝게도 시간관계상 Task1만 마무리하였습니다. 초기 데이터 및 서비스 분석과 연휴기간이 겹처 Task1&2를 둘다 만드는데에는 시간이 부족했습니다.(재출 이후에도 진행할 예정입니다.)

Task1의 경우 아래와 같은 결과가 나왔습니다. 로컬환경(MacBook Air m2)에서 cpu환경에서 테스트르 진행하였습니다.

### 생성 결과

| Example                                                                                                                                        | Time     | Result                          |
|---------------------------------------------------------|----------|---------------------------------|
| { "experience": 1, "gender": "남성", "purpose": 1, "rm_weight": { "bench_press": 40, "squat": 50, "dead_lift": 55, "over_head_press": 30 } }   | 43.26sec | ![test](/static/ex1_result.png) |
| { "experience": 2, "gender": "여성", "purpose": 3, "rm_weight": { "bench_press": 60, "squat": 70, "dead_lift": 75, "over_head_press": 40 } }   | 42.82sec | ![test](/static/ex2_result.png) |
| { "experience": 3, "gender": "남성", "purpose": 2, "rm_weight": { "bench_press": 80, "squat": 100, "dead_lift": 105, "over_head_press": 45 } } | 44.94sec | ![test](/static/ex3_result.png) |

```shell
Example1
{
  "program": "5/3/1 프로그램",
  "init_weight_rate": 55,
  "increase_rate_week": 10,
  "increase_rate_set": 5,
  "deloading_rate": 50,
  "weekly_weight_increase_plan": "Week1\nSET1 55%x5\nSET2 60%x5\nSET3 65%x5+\n\nWeek2\nSET1 60%x3\nSET2 65%x3\nSET3 70%x3+\n\nWeek3\nSET1 65%x5\nSET2 75%x3\nSET3 85%x1+\n\nWeek4\nSET1 50%x5\nSET2 55%x5\nSET3 60%x5\n"
}

Example2
{
  "program": "5/3/1 프로그램",
  "init_weight_rate": 65,
  "increase_rate_week": 10,
  "increase_rate_set": 5,
  "deloading_rate": 40,
  "weekly_weight_increase_plan": "Week1\nSET1 65%x5\nSET2 75%x5\nSET3 85%x5+\n\nWeek2\nSET1 70%x3\nSET2 80%x3\nSET3 90%x3+\n\nWeek3\nSET1 75%x5\nSET2 85%x3\nSET3 95%x1+\n\nWeek4\nSET1 40%x5\nSET2 50%x5\nSET3 60%x5\n"
}

Example3
{
  "program": "5/3/1 프로그램",
  "init_weight_rate": 55,
  "increase_rate_week": 10,
  "increase_rate_set": 5,
  "deloading_rate": 50,
  "weekly_weight_increase_plan": "Week1\nSET1 55%x5\nSET2 60%x5\nSET3 65%x5+\n\nWeek2\nSET1 60%x3\nSET2 65%x3\nSET3 70%x3+\n\nWeek3\nSET1 65%x5\nSET2 75%x3\nSET3 85%x1+\n\nWeek4\nSET1 50%x5\nSET2 55%x5\nSET3 60%x5\n"
}
```
