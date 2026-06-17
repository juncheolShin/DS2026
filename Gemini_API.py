import json
import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 17개 속성 정의
STEAM_ASPECTS = [
    '최적화#프레임', '최적화#조작감', '시스템#버그', 
    '콘텐츠#볼륨', '콘텐츠#몰입도', '콘텐츠#난이도', '콘텐츠#피로도', '시스템#밸런스', '시스템#독창성', '시스템#자유도', '시스템#진입장벽', 
    'UX#그래픽', 'UX#캐릭터디자인', 'UX#사운드', '스토리#내러티브', '운영#핵/치트', '운영#업데이트'
]

# GPT용 프롬프트 및 가이드라인 반영 (개정본)
SYSTEM_PROMPT = f"""
너는 스팀(Steam) 게임 리뷰를 분석하여 속성 기반 감성 분석(ABSA) 학습용 초안 데이터를 생성하는 전문 데이터 엔지니어다.
입력되는 스팀 리뷰 텍스트를 분석하여 [속성 카테고리, 감성] 쌍을 추출하고, 지정된 JSON 포맷으로만 출력해라.

[허용된 속성 카테고리 목록]
{json.dumps(STEAM_ASPECTS, ensure_ascii=False)}
* 이 리스트 외의 단어는 절대 카테고리 이름으로 사용하지 마라. 상위#하위 구조를 완벽히 유지해라.

[속성별 상세 정의 및 판단 키워드]
1. 최적화#프레임: FPS, 렉, 버벅임, 프레임 드랍, 팅김, 강제종료 등 실행 성능 관련
2. 최적화#조작감: 컨트롤, 입력 반응성, 키보드/마우스/패드 조작 편의성, 손맛
3. 시스템#버그: 게임 내 오류, 꼬임 현상, 시스템 충돌, 데이터 파손 등 예상치 못한 오작동
4. 콘텐츠#볼륨: 콘텐츠의 양, 플레이타임(플탐), 분량, 즐길 거리의 많고 적음
5. 콘텐츠#몰입도: 재미, 중독성, 시간 순삭, 갓겜, 계속하게 만드는 매력
6. 시스템#밸런스: 캐릭터/아이템 능력치, 매칭 시스템의 균형, 너프/버프 관련
7. 시스템#독창성: 차별화된 시스템, 독특함, 유니크함, 처음 보는 메커니즘
8. 시스템#자유도: 플레이어의 선택지, 빌드 커스텀, 높은 자유도
9. UX#그래픽: 비주얼 품질, 아트 스타일, 텍스처, 화면 및 맵 디자인
10. UX#캐릭터디자인: 캐릭터 외형, 모델링, 스킨, 일러스트
11. UX#사운드: 배경음악(BGM), 효과음, 음향 품질 및 타격음
12. 스토리#내러티브: 세계관, 시나리오, 서사, 퀘스트 스토리의 품질
13. 운영#핵/치트: 불법 프로그램, 매크로, 봇, 어뷰징, 치터 대응 관련
14. 운영#업데이트: 패치 주기, 업데이트 내용, 핫픽스, 소통 및 방향성
15. 콘텐츠#난이도: 게임 자체의 시스템적 어려움 수준, 보스 패턴의 하드코어함 등
16. 시스템#진입장벽: 뉴비/초보자가 게임 시스템을 배우고 적응하기 어려운 정도 (고인물 문제 등)
17. 콘텐츠#피로도: 반복 플레이(노가다), 지루함, 숙제 같은 플레이 요소로 인한 피로

[감성 분류 기준]
- positive: 해당 속성에 만족, 찬사, 추천하는 경우
- negative: 해당 속성에 불만, 비판, 아쉬움을 표하는 경우 (렉, 버그, 핵, 진입장벽 등의 문제 언급 포함)
- neutral: 감정 없이 객관적 사실만 기술하거나, 판단을 보류한 경우

[예외 처리 및 판정 규칙 (Edge Cases)]
- **난이도 vs 진입장벽**: 게임 플레이 자체의 어려움은 '콘텐츠#난이도'로, 시스템이 복잡하여 초보자가 배우기 어려운 점은 '시스템#진입장벽'으로 분류한다. (예: "초보는 하지 마세요" -> 시스템#진입장벽: negative)
- **몰입도 vs 피로도**: 재미있고 중독적이라는 표현은 '콘텐츠#몰입도'로, 반복 노가다가 심하고 지루하다는 표현은 '콘텐츠#피로도'로 분류한다.
- **상쇄 및 중립 처리**: 하나의 리뷰 안에서 동일한 속성에 대해 긍정 표현과 부정 표현이 모두 명확히 등장하는 경우(예: "그래픽은 좋은데 배경 아트는 별로다"), 해당 속성은 `neutral`로 판단한다.
- **리뷰 문맥 우선**: 욕설/비속어가 포함되어 있어도 문맥을 파악해 라벨링하며, 풍자나 반어법은 전체 맥락을 보고 실제 의도를 파악하라. 특정 측면에 대한 직접적인 언급 방향을 최우선으로 한다.

[출력 형식 제약 조건]
결과는 마크다운(```json 등) 없이 오직 아래 구조를 가진 순수 JSON 객체(Dict)로만 출력해라.
한 문장에 여러 속성이 보이면 annotation 배열 내에 복수로 추출해라. 만약 분석된 속성이 하나도 없다면 빈 배열([])로 두어라.

[출력 JSON 예시]
{{
  "sentence_form": "리뷰 문장 원문",
  "annotation": [
    ["최적화#프레임", "negative"],
    ["콘텐츠#난이도", "positive"]
  ]
}}
"""

# Few-shot 예시: 사람이 직접 작성하여 품질 기준을 잡아주는 고품질 샘플 쌍
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "카운터 스트라이크는 둠을 잇는 거대 명작 게임이다. FPS의 본질인 속도, 에임, 생존이라는 철학을 전술과 팀 구조로 완성해낸 작품이다. 나는 둠 플레이어였다. 도스 시절 둠에서 손과 눈으로 싸우는 감각을 익혔고, 그 감각은 카운터 스트라이크 1.6에서 팀과 라운드, 경제라는 구조로 이어졌으며, 지금은 카운터 스트라이크 2까지 자연스럽게 이어지고 있다. 둠이 끝없이 밀어붙이는 생존 시나리오였다면, 카운터 스트라이크는 그 생존을 사람 대 사람의 경쟁으로 압축한 진화형 명작이다."},
    {"role": "assistant", "content": json.dumps({
        "sentence_form": "카운터 스트라이크는 둠을 잇는 거대 명작 게임이다. FPS의 본질인 속도, 에임, 생존이라는 철학을 전술과 팀 구조로 완성해낸 작품이다. 나는 둠 플레이어였다. 도스 시절 둠에서 손과 눈으로 싸우는 감각을 익혔고, 그 감각은 카운터 스트라이크 1.6에서 팀과 라운드, 경제라는 구조로 이어졌으며, 지금은 카운터 스트라이크 2까지 자연스럽게 이어지고 있다. 둠이 끝없이 밀어붙이는 생존 시나리오였다면, 카운터 스트라이크는 그 생존을 사람 대 사람의 경쟁으로 압축한 진화형 명작이다.",
        "annotation": [["콘텐츠#몰입도", "positive"], ["시스템#독창성", "positive"]]
    }, ensure_ascii=False)},
    {"role": "user", "content": "재미있어요 초보가 하기에는 어렵긴 한데 재밌음 이런류 게임 자체를 처음 해봐서 애임도 안 좋고 반응 속도도 느리지만 친구들이랑 하면 재밌어요 ㅎㅎ"},
    {"role": "assistant", "content": json.dumps({
        "sentence_form": "재미있어요 초보가 하기에는 어렵긴 한데 재밌음 이런류 게임 자체를 처음 해봐서 애임도 안 좋고 반응 속도도 느리지만 친구들이랑 하면 재밌어요 ㅎㅎ", 
         "annotation": [["콘텐츠#몰입도", "positive"], ["시스템#밸런스", "negative"], ["시스템#진입장벽", "negative"]]
    }, ensure_ascii=False)},
    {"role": "user", "content": "농장 시뮬겜 하면 떠오르는 대표작 시작하자마자 보이는 황당한 주민 얼굴은 무시하세요 어차피 다 잘생기고 예쁜 리텍으로 성형 시켜서 하니깐 ㄱㅊ 아요 전혀 문제 되지 않슴다 계절마다 이벤트도 많고 노가다 할 것도 많아요 마을 npc랑 미연시도 해야 하고 진짜 하다 보면 시간 훅 가 있는데 아직 절반도 안 왔다는 점 이런 류 게임 좋아하는 사람이라면 돈이 전혀 아깝지 않은 게임입니다"},
    {"role": "assistant", "content": json.dumps({
        "sentence_form": "농장 시뮬겜 하면 떠오르는 대표작 시작하자마자 보이는 황당한 주민 얼굴은 무시하세요 어차피 다 잘생기고 예쁜 리텍으로 성형 시켜서 하니깐 ㄱㅊ 아요 전혀 문제 되지 않슴다 계절마다 이벤트도 많고 노가다 할 것도 많아요 마을 npc랑 미연시도 해야 하고 진짜 하다 보면 시간 훅 가 있는데 아직 절반도 안 왔다는 점 이런 류 게임 좋아하는 사람이라면 돈이 전혀 아깝지 않은 게임입니다", 
         "annotation": [["콘텐츠#볼륨", "positive"], ["콘텐츠#몰입도", "positive"]]
    }, ensure_ascii=False)}
]

def request_gpt_labeling(review_text):
    """단일 리뷰 텍스트를 GPT API에 던져 ABSA 초안 JSON을 받아오는 함수"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": review_text})
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",  # 비용 효율성이 가장 좋은 gpt-4o-mini 모델 추천
            messages=messages,
            temperature=0.0,      # 결과의 일관성과 객관성을 위해 온도를 0으로 고정
            response_format={"type": "json_object"} # JSON 출력 강제
        )
        
        # GPT 출력을 파싱
        result_json = json.loads(response.choices[0].message.content)
        # 만약 GPT가 리스트가 아닌 단일 오브젝트 구조로 감쌌을 경우를 대비한 안전 장치
        if isinstance(result_json, dict) and "sentence_form" not in result_json:
            return list(result_json.values())[0]
        return result_json
        
    except Exception as e:
        print(f"Error processing review: {review_text[:20]}... -> {e}")
        return {"sentence_form": review_text, "annotation": []}

# 실전 가동 및 파일 저장 파이프라인
if __name__ == "__main__":
    # 수집한 스팀 리뷰 데이터 리스트 (실제 데이터프레임 컬럼 연동 가능)
    csv_path = "reviews_raw.csv"

    try:
        # 표준 UTF-8로 시도 후, 에러 발생 시 한국어 윈도우 엑셀 포맷(CP949)으로 재시도
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    
    target_column = "review" if "review" in df.columns else df.columns[0]
    raw_steam_reviews = df[target_column].dropna().astype(str).tolist()
    
    output_dataset = []
    
    print("GPT 보조 프리 라벨링 시작...")
    for idx, review in enumerate(raw_steam_reviews):
        print(f"[{idx+1}/{len(raw_steam_reviews)}] 변환 중...")
        gpt_output = request_gpt_labeling(review)
        
        if isinstance(gpt_output, dict) and "sentence_form" in gpt_output:
                gpt_output["id"] = f"steam_{str(idx+1).zfill(5)}"
                output_dataset.append(gpt_output)
            
    # 5. 최종 데이터셋 파일 저장 및 구조 확인
    output_path = "./steam_reviews_pre_labeled.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for data in output_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"\n프리 라벨링 완료! 저장 경로: {output_path}")
    
    # 예외 처리: 데이터가 안전하게 담겼을 때만 샘플 출력
    if len(output_dataset) > 0:
        print("저장된 첫 번째 샘플 데이터 구조 확인:")
        print(json.dumps(output_dataset[0], indent=2, ensure_ascii=False))
    else:
        print("경고: 조건에 맞는 데이터가 추출되지 않아 리스트가 비어 있습니다. 프롬프트 출력을 점검하세요.")
