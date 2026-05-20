# AI·빅데이터 분석 기반 보훈심사 지원 시스템 — 연구동향 조사 보고서

**발주처**: 국가보훈부 / 한국지능정보사회진흥원(NIA)
**사업예산**: 1,797,452,497원 (부가세 포함)
**사업기간**: 계약체결일로부터 180일
**조사일**: 2026-05-20

---

## 1. 제안요청서 핵심 요약

### 1.1 사업 목표
보훈심사(요건심사 + 상이등급 심사)의 자동화·지능화 전환을 통해 **현재 평균 5개월 이상 소요되는 보훈심사 기간을 1차년도 120일 → 3차년도 70일까지 단축**하고, 담당자 1인당 월평균 처리건수를 40건 → 70건으로 75% 증가시키는 것이 핵심 KPI.

### 1.2 5대 핵심 기능 요구사항 (SFR)

| 번호 | 기능 | 핵심 기술 |
|---|---|---|
| SFR-001~003 | 입증자료 OCR/VLM 디지털화 + 필수정보 자동 추출 (20여개 항목) | VLM, OCR, NLP, 의학·법률·한자 전문용어 인식 |
| SFR-004 | 지능형 지식·참고자료 자연어 검색 (법령·매뉴얼·판례·Q&A) | RAG, LLM, 대국민 챗봇 확장 |
| SFR-005 | 의미 기반 과거 유사 심사사례 매칭·추천 + 판단근거·쟁점 도출 | Vector DB, RAG, 의미 유사도 |
| SFR-006 | 데이터 기반 심사결과 예측 및 근거 제시 (상이등급, 요건 해당여부) | Legal Judgment Prediction (LJP) + XAI |
| SFR-007 | 표준 서식 심사 검토서 자동 생성 + 출처 위치 정보 | LLM 생성, RAG grounding |

### 1.3 데이터 자산
- 심의의결서 **468,000건** (DB, 텍스트)
- 심의의결 근거자료 **1,600,000건** (이미지, ~12TB)
- 보훈민원 상담 **22만건** (DB)
- 업무지침/민원편람/FAQ ~30권 (이미지), 외부 판례 자료

### 1.4 품질·평가 핵심 기준
- **TER-004 (RAG 검색·응답 품질)**: F1, Recall, Precision, BLEU, ROUGE 메트릭. **테스트케이스 90% 이상 정답률** 달성 필수
- **DAR-010 (RAG 데이터셋 값 검증)**: VLM(AI-OCR) 성능 검증, VectorDB 값 검증, 서비스 의미 검증
- **PER-002 (응답시간)**: 일반 기능 3초 이내
- **PER-005 (동시접속)**: 기준 사용자 수 100% 동시접속시에도 비례하지 않는 비정상적 지연 없음
- **TER-006 (통합테스트)**: 결함율 5% 이상시 재실시 요구 가능

### 1.5 인프라 제약
- 민감 의료·개인정보로 인해 **외부망과 분리된 폐쇄망 운영**, 클라우드 외부 API 활용 제한적
- 범정부 AI 공통기반 활용, RAG는 **민관협력형 클라우드(PPP)존**에 별도 저장
- 범정부 AI 공통기반 임베딩 모델 또는 제안사 모델 사용 가능 (탑재 가능해야 함)

### 1.6 RFP 추진전략 7대 기둥 (사업 정체성 정의)

RFP 3장 추진계획 / 나. 추진전략에 명시된 7대 추진 방향. 평가위원이 가장 중요시하는 사업 이해도 평가의 기준선:

1. **하이브리드 보안 아키텍처** — 범정부 AI 공통기반(민간LLM) + PPP존 RAG 분리
2. **빅데이터 기반 심사 인프라** — OCR/VLM·RAG·유사사례·예측·검토서 통합
3. **데이터 품질 제고** — VLM OCR + 전문 인력 단계별 정제
4. **도메인 특화 학습** — RAG DB + 프롬프트 최적화로 보훈 도메인 정확도 확보
5. **휴먼 인 더 루프** — **"AI는 보조도구, 최종 판단은 보훈심사위원회. 의사결정 지원 도구로서의 위상 명확화"** (본 보고서 5장 직접 대응)
6. **AI 도입 관련 제도 정비** — 유공자법 AI 활용근거 명시, 개인정보보호 내부 지침 정비
7. **서비스 확장 및 기관간 협력** — 표준 API 체계, 인사처·국방부·고용부·복지부 등 연계

> 본 보고서의 핵심 차별화 5대 축(2장 아키텍처, 3장 품질평가, 4장 차별화 종합, 5장 HITL, 6장 운영지속)이 위 7대 추진전략을 정면 흡수하도록 설계되었다.

---

## 2. 프로젝트 목표·기능 관련 연구동향

### 2.1 OCR/VLM 기반 의료·법률문서 디지털화 (SFR-001~003 대응)

#### (1) Vision-Language Model (VLM)의 부상

전통적 OCR 파이프라인(이미지 → OCR → 텍스트 정제 → NLP → 정보추출)은 **여러 단계의 오류 누적, 문맥 손실, 유지보수 비용** 문제가 있다. 2025년 들어 End-to-End VLM이 사실상 표준으로 자리잡고 있다 (TechEon, 2025; Ubicloud, 2025).

대표 오픈 VLM 모델 (2025년 기준):
- **DeepSeek-OCR** (MIT 라이선스, 고처리량 배치 최적화)
- **Qwen2.5-VL-7B** (다국어 OCR + 의미 이해)
- **Chandra** (의료기록·법률문서 등 특화)
- 한국어 환경에서는 **Naver HyperCLOVA X** 기반 멀티모달 활용이 법률 자문 시스템에서 검증되고 있음 (Continental Aju 법률 Q&A, PEFT + RAG)

#### (2) 의료문서 정보추출 사례

- **Fullerton Health**(2025) — PaddleOCR + Logistic Regression 분류기 + Qwen2.5-VL-7B 하이브리드 파이프라인. **문서타입 분류 95% 정확도, 필드 추출 87% 정확도, 평균 2초 미만 지연**. 본 사업의 "건당 20~30종 입증자료, 20여개 필수 항목 자동추출"과 매우 유사한 설정
- **Valerie Health** — 스캔/팩스/수기 양식을 통합 API로 처리, EMR 직매핑, HIPAA 준수 자동화 (firstsource.com 사례)
- **Patchfinder** (arXiv 2412.02886) — VLM의 **모델 불확실성을 활용한 정확도 부스팅**. 4.2B 파라미터로 93.3% 정확도 달성 (vs. OCR+Phi-3-mini 66.7%)

#### (3) 미국 보훈청(VA) 직접 사례 — 가장 직접적 참고

가장 직접적으로 본 사업과 유사한 운영 사례. <https://meritalk.com/articles/how-the-va-leveraged-ai-to-accelerate-claims-processing/> 및 The War Horse(2025):

- VA + Booz Allen Hamilton 공동 구축: **Amazon Textract 기반 Intelligent Document Processing(IDP) 파이프라인**
- **재향군인 1인당 평균 100개 문서, ~1,300페이지의 의료증거·복무이력 처리** (본 사업 1인당 40건/월 → 60건/월 목표와 유사 구조)
- 2025 회계연도 사상 최대 **3,001,734건 장애보상·연금 신청 처리**
- 우편 처리시간 **10일 → 0.5일**로 단축
- **Smart Ratings Recommendation** (2025년 12월 배포 예정) — 단순 증거 요약을 넘어 등급(rating) 추천까지 제공하는 단계. 본 사업 SFR-006와 정확히 같은 방향성
- **AICES (AI Claims Evaluation System)** — Acceptable Clinical Evidence(ACE)를 활용하여 대면 검사를 최소화

**중요 시사점**: VA 사례에서 "Hit or Miss" 평가, 변호사들이 AI 누락 여부를 수동 확인해야 한다는 비판이 있음. 따라서 본 사업도 **Human-in-the-loop 검증 메커니즘 + 누락 탐지 메트릭 설계**가 결정적임.

#### (4) 의료 OCR의 보안 이슈

**PHI Leakage** 문제: 비전 토큰 마스킹만으로는 의료문서의 PHI(Protected Health Information) 유출을 막을 수 없다는 최근(2025) 연구 (arXiv 2511.18272). 본 사업의 폐쇄망 운영 요구사항과 직결되는 이슈로, **OCR 처리 중간단계의 로그·메모리에 잔존하는 민감정보 통제 방안**까지 제안서에 반영 필요.

---

### 2.2 법률·의료 도메인 RAG (SFR-004 대응)

#### (1) Legal RAG 종합 동향

- **Hindi et al. (2025) IEEE Access** — "Enhancing the precision and interpretability of RAG in legal technology: A survey" — 법률 RAG의 메트릭(Precision, Recall, MRR, MAP) 체계화
- **CLERC** (arXiv 2406.17186) — 법률 사례 검색 + RAG 분석생성용 데이터셋
- **LegalBench-RAG** (arXiv 2408.10343) — 법률 도메인 RAG 벤치마크
- **Zheng et al. (CS&Law 2025)** — Bar Exam QA + Housing Statute QA 벤치마크. **BM25와 현재의 dense retriever 모두 법률 도메인에서 gold passage 식별 능력이 제한적**임을 보여줌

#### (2) Hybrid Retrieval (BM25 + Dense + Reranker)

본 사업의 폐쇄망 환경과 OSS LLM(`gpt-oss-120b` 등) 활용 가정에서 가장 실용적 패턴:

- **HyPA-RAG** (NAACL 2025 Industry) — 쿼리 복잡도 분류기로 파라미터 적응형 RAG. 법률 도메인(NYC Local Law 144)에서 검증
- **Reciprocal Rank Fusion (RRF)** (Cormack 2009) — 어휘·의미 검색 결과 융합의 사실상 표준
- **Self-RAG, HyDE** — 검색 필요성 판단, 가상 답변 임베딩 활용
- 47billion 분석(2026) — Neo4j + LlamaIndex 하이브리드로 **multi-hop 정밀도 3배 향상**, 법률 감사가능성 강화

#### (3) Knowledge Graph + RAG (Hybrid KG-RAG)

본인의 한국 세법 시스템 작업과 직접 연결되는 부분:

- **Domain-Partitioned Hybrid RAG** (arXiv 2602.23371, 2025) — 인도 법률 시스템. **3개 RAG 파이프라인(판례, 성문법, 형법) + Neo4j Legal KG**, LLM agentic orchestrator. **하이브리드 70% pass rate vs. RAG-only 37.5%** — 약 2배 성능 차이
- **Benchmarking KG-based RAG** (CEUR-WS Vol-4079, 2025 RAGE-KG workshop) — HippoRAG 2, Nano GraphRAG, LightRAG, LlamaIndex 비교. 법률 QA 벤치마크
- **Microsoft GraphRAG** — community 계층 사전요약으로 복잡 법률 작업 **20–70% 향상**

본 사업에서는 5개 CaseType(심판/판례/예규/지방세판례/지방세예규에 상응하는 보훈심사 분류) × 법령·매뉴얼·판례 다층 구조이므로, **세법 도메인의 Hybrid KG-RAG 아키텍처가 거의 그대로 이식 가능**.

#### (4) 한국 법률 NLP 동향

- **HyperCLOVA X** — 한국어 법률·통계 자연어 처리에 최적화, 환각률이 영어 LLM 대비 낮음 (Nucamp, 2025)
- 본 사업 명시된 **범정부 AI 공통기반** — 행정안전부 주도로 표준화된 임베딩·LLM 모델 제공
- 한국어 도메인 임베딩 모델: KoSimCSE, BGE-M3, ko-sroberta-multitask 등

---

### 2.3 의미 기반 유사 심사사례 검색 (SFR-005 대응)

본 사업에서 **47만건의 심의의결서**를 대상으로 의미적으로 유사한 과거 사례를 추천하고 판단근거·쟁점을 도출하는 기능은, 법률 IR에서 가장 활발히 연구되는 영역.

#### (1) Legal Case Retrieval (LCR) 핵심 연구

- **COLIEE 2021–2025** — 법률 사례 검색·entailment 국제 대회. **노력의 베이스라인 표준**
- **DoSSIER@COLIEE 2021** (arXiv 2108.03937) — **BM25 + Dense Retrieval (단락 단위) + BERT 재순위**, 도메인 특화 임베딩 사용. 본 사업의 단락 단위 청킹·하이브리드 검색 설계에 직접 적용 가능
- **UQLegalAI@COLIEE 2025** (arXiv 2505.20743) — LLM + GNN 결합. 인용 네트워크를 GNN으로 학습
- **ReaKase-8B** (arXiv 2510.26178, 2025) — 법률 사례의 **핵심 구성요소 추출 + 추론 + 지식 트리플릿 생성**으로 임베딩 향상. 본 사업의 "판단근거·쟁점 자동도출" 요구사항과 직접 부합
- **EMV (Event Multi-View)** (Zhang et al., Information Processing & Management) — Llama3-8B로 사건체인·참여자역할 추출. 텍스트 유사도 + event-chain 유사도 + event-type overlap + role-aware alignment **4개 신호를 learnable aggregator로 결합**

#### (2) 비용·정확도 트레이드오프

- **Mentzingen et al. (2025)** Artificial Intelligence and Law — 브라질 행정법원 데이터. POS 기반 요약 + ADA 임베딩 조합이 비용·성능 최적 트레이드오프. 본 사업의 **PPP존 비용 제약과 동시접속 처리 요구**에 시사점 큼.

#### (3) 구조 정보 활용

- **Incorporating Structural Information into Legal Case Retrieval** (ACM TOIS, 2023) — 법률 사례는 길고 복잡한 구조. **사례 내·외부 구조를 모두 활용한 learning-to-rank** 모델. 본 사업의 의결서 표준 서식(기본서식 존재)을 활용한 구조 기반 청킹의 근거.

---

### 2.4 심사결과 예측 + 판단근거 제시 (SFR-006~007 대응)

#### (1) Legal Judgment Prediction (LJP)

- **PredEx** (arXiv 2406.04136) — 인도 법률 도메인, **약 15,000건 annotated 데이터**. 단순 결과 예측을 넘어 **설명까지 제공**하는 instruction-tuning 데이터셋. 본 사업의 의결서 47만건 학습 시 annotation 전략 참고 가치
- **LexFaith-HierBERT** (Nature Scientific Reports, 2026) — 계층적 BERT + **관계 rationale head + faithfulness-aware attention**. SHAP/LIME + 모델 자체 attention saliency map으로 transparency 확보
- **Knowledge Fusion + Dependency Masking** (PLOS ONE, 2026-01-16) — 법률 조문, 죄목, 형량의 의존성 마스킹. 다중 subtask 간 의존성 처리

#### (2) Explainable AI (XAI) for Legal AI

- **Explainable AI and Law: An Evidential Survey** (Digital Society, Springer 2023) — intrinsic 방법(linear, decision tree, rule-based) vs. post hoc 방법(SHAP, LIME 등) 비교
- 본 사업 RFP의 "판단근거 제공", "유사사례 기반 등급 분포 분석", "예측 결과에 대한 판단 근거 제공"은 정확히 XAI 요구사항. **SHAP/LIME 같은 post-hoc 설명 + RAG의 출처(원데이터 위치) 표시**의 이중 설명 체계가 표준

#### (3) Hallucination/환각 문제 — 가장 결정적 리스크

- **Magesh et al. (2024)** — 미국 상용 법률 RAG 시스템(LexisNexis, Westlaw 등) 분석. **가장 우수한 시스템도 17% 환각률**. 본 사업은 보훈대상자의 권리에 직결되는 결정이므로, 17% 환각률은 받아들일 수 없는 수준
- **AILA hallucinations claim** — AI 법률 도구들이 환각 방지 능력을 과장하고 있다는 실증연구 (Huang et al., 2023)
- VA에서도 변호사들이 **누락 여부를 수동 확인**하는 실태 — Human-in-the-loop의 본질적 필요성

---

## 3. 품질 관리 / 성능 평가 관련 연구동향

본 사업 TER-004, DAR-010, QUR-001~011, PER-001~006의 정량 검증 체계를 어떻게 설계할지의 근거.

### 3.1 RAG 평가 프레임워크 — 사실상 표준

#### (1) RAGAS (Retrieval-Augmented Generation Assessment)

- **EACL 2024** 공식 발표, GitHub `explodinggradients/ragas`. **2024년 기준 월 500만 평가 처리**, AWS·Microsoft·Databricks·Moody's 등이 운영
- 4대 핵심 메트릭:
  - **Faithfulness** — 응답이 검색된 컨텍스트에 grounding되는가 (환각 측정)
  - **Answer Relevancy** — 응답이 질문에 관련 있는가
  - **Context Precision** — 검색된 컨텍스트가 관련 있는 것 중심으로 정렬되었는가
  - **Context Recall** — 정답에 필요한 컨텍스트를 모두 검색했는가
- **Reference-free** 평가 가능 — 본 사업의 도메인 전문가 검증 비용 절감에 결정적

#### (2) DeepEval (Confident AI)

- `confident-ai/deepeval` — 오픈소스. **5줄 코드로 SOTA RAG 메트릭 구현**
- Retriever: Contextual Recall/Precision/Relevancy (top-K 튜닝, 임베딩 모델 선정용)
- Generator: Faithfulness, Answer Relevancy (LLM·프롬프트 평가용)
- Custom metric, golden dataset 합성 지원
- **Ragas vs. DeepEval 비교 (2025)**: RAGAS는 strict entailment로 사실 불일치 잘 잡고, DeepEval은 미묘한 잘못된 표현·의도 수준 이슈 포착. **두 가지를 모두 사용하는 것이 권장**

#### (3) 전용 환각 탐지 모델

- **HHEM (Hughes Hallucination Evaluation Model)** — Vectara의 분류 모델. RAGAS의 LLM-as-a-judge보다 신뢰성·강건성 우수. 오픈소스 공개
- **Lynx** — HaluBench 벤치마크에서 RAGAS Faithfulness 능가, 특히 long context에서 우수
- **Patronus AI** — 실시간 모니터링·시각화 대시보드, conciseness/bias 등 부가 평가자

### 3.2 정보 검색(IR) 평가 메트릭 — 본 사업 TER-004 명시 메트릭

본 사업 TER-004가 명시한 **F1, Recall, Precision, BLEU, ROUGE**는 IR + 생성 평가의 결합 형태. 보완 필요 메트릭:

| 영역 | 메트릭 | 의미 |
|---|---|---|
| Retrieval | Precision@k, Recall@k | top-k 결과의 정밀도·재현율 |
| Retrieval (Ranking) | **MRR** (Mean Reciprocal Rank) | 첫 정답 위치의 역수 평균 |
| Retrieval (Ranking) | **MAP** (Mean Average Precision) | 각 쿼리 AP의 평균 |
| Retrieval (Ranking) | **nDCG** (normalized DCG) | 위치 가중 누적 이득 |
| Generation | BLEU, ROUGE-1/2/L | n-gram·LCS 기반 텍스트 일치도 |
| Generation | BERTScore | 임베딩 기반 의미 유사도 |
| Generation (Domain) | Legal Faithfulness | 법률 진술의 충실도 (LexFaith) |

본 사업 제안서에서는 **RFP 명시 메트릭(F1/Recall/Precision/BLEU/ROUGE)을 기본**으로 하되, **RAGAS의 Faithfulness + Answer Relevancy + Context Recall, MRR/nDCG**를 추가 제안하여 차별화를 둘 수 있음.

### 3.3 골든 데이터셋 구축 방법론

본인이 최근 작업 중인 **세법 도메인 골든 데이터셋 평가 패키지** 경험과 직결:

- **테스트케이스 50개 이상** (RFP TER-004 명시) — 데이터 중 정답이 존재하는 문항 50개 이상 입력 후 결과로 정확도 확인
- **DecisionType 의미 그룹화, 정관 버전 선정 정책, 질의 유형 카테고리화** 같은 다층 구조의 도메인 전문가 review가 필요
- LegalBench (NeurIPS 2024) — 162개 법률 추론 태스크 수집
- COLIEE — 매년 갱신, retrieval/entailment 분리 평가

### 3.4 시스템 성능 평가

- 동시접속자, 응답시간, 처리량 등 전통적 비기능 메트릭은 **k6, JMeter, Locust** 등으로 검증
- LLM 시스템 특유의 메트릭:
  - **Time to First Token (TTFT)**, **Tokens per Second (TPS)**
  - **vLLM, TensorRT-LLM, SGLang** 등 추론 서버 벤치마킹
  - PPP존 GPU 자원 제약 하에서 OSS LLM(`gpt-oss-120b`, Gemma3 등) 운영 시 quantization (AWQ, GPTQ, FP8) 트레이드오프

### 3.5 데이터 품질 (DAR-010 대응)

본 사업이 명시한 **3단계 데이터 검증 체계**:

1. **VLM(AI-OCR) 성능 검증** — character/word/field-level accuracy, F1
2. **VectorDB 값 검증** — 임베딩 차원, 무결성, 중복, 메타데이터 일관성
3. **RAG 서비스 의미 검증** — RAGAS/DeepEval로 end-to-end

연구 동향상 주목 가치 있는 데이터 품질 프레임워크:

- **Microsoft Presidio** — PII/PHI 탐지·마스킹 오픈소스. 본 사업의 민감정보 처리에 직접 필요
- **Great Expectations / Soda** — 데이터 파이프라인 expectation 테스트
- **Deepchecks** — ML 모델·데이터 검증

---

## 4. 본 사업 제안서 차별화 방향 제언

> 본 장은 RFP 명시 요구사항의 베이스라인 구축 관점 차별화 포인트를 정리한다. **(1) HITL 협업체계는 RFP 추진전략으로 명시된 비중을 반영해 5장에서**, **(2) 운영 단계 catastrophic forgetting / data shift 대응 메커니즘은 6장에서** 각각 독립 본론으로 다룬다.

### 4.1 RFP 요구사항을 넘어선 기술 차별화 포인트

#### (1) Hybrid KG-RAG 아키텍처 강조
- RFP는 RAG(벡터DB) 중심이지만, **47만건 의결서 + 법령 + 판례 간 관계**는 그래프 구조로 표현했을 때 multi-hop 추론·인용 추적이 가능
- 본인의 세법 도메인 경험 (Hybrid Neo4j + Qdrant/pgvector) 그대로 이식
- 인도 법률 사례 (Domain-Partitioned Hybrid RAG, 70% vs 37.5%)를 근거로 효과성 입증

#### (2) Agentic Orchestrator (LLM Router)
- 쿼리 라우팅(VDB / GDB / VDB→GDB / GDB→VDB) — 본인 작업 중인 `check_q.py`, `hybrid_router.py` 패턴
- 3-tier routing (Gate → rule-based → LLM)
- few-shot 27개 패턴 같은 OSS LLM 친화적 설계

#### (3) 다층 환각 방지 + HITL (5장 본론에서 상세)
- Self-RAG 스타일 self-check
- HHEM/Lynx로 사후 검증
- 출처 원데이터 위치 명시 (RFP SFR-007 요구)
- **HITL 협업체계** — RFP가 추진전략으로 명시한 "사람 중심 AI 협업". Confidence 기반 Learning-to-Defer + 신속/보통/심화 트랙별 HITL 강도 차별화. 자세한 내용은 5장 참조

#### (4) VLM End-to-End 입증자료 처리
- Qwen2.5-VL 또는 Chandra 같은 도메인 친화 VLM (RFP는 OCR/VLM 처리 SW를 자체 도입 인정 — 본 사업비에 GPU 등 자원 포함)
- VA-Booz Allen-Textract 사례를 벤치마크로 제시
- 의학용어·한자·서식 인식을 위한 VLM 도메인 적응 (PEFT/LoRA 적용 가능 — VLM에 한해. 범정부 AI 공통기반 LLM은 동결)

### 4.2 품질 관리 차별화 포인트

#### (1) 4계층 평가 체계 (TER-004 / DAR-010 발전형)

```
L1 VLM OCR Layer:   Character/Word F1, BLEU, layout F1
L2 Retrieval Layer: Precision@k, Recall@k, MRR, nDCG, Context Recall/Precision
L3 Generation Layer: BLEU, ROUGE-L, Faithfulness, Answer Relevancy, HHEM hallucination
L4 Domain Layer:    Legal Faithfulness, 도메인 전문가 평가 (Inter-Annotator Agreement)
```

#### (2) Drift Monitoring 4계층 (6장 본론에서 상세)
본인이 의성군 제안에서 정립한 4계층 운영 모니터링 — L1 Infrastructure / L2 Retrieval Quality / L3 Response Quality / L4 Business Quality. 보훈심사 KPI(쟁송율·심사 일관성·적체 트렌드)에 맞춰 L4 메트릭을 재정의하여 그대로 적용. 자세한 내용은 6.2절 참조

#### (3) Golden Dataset 표준화
- 도메인 전문가 검증 프로토콜
- IAA (Cohen's κ, Fleiss' κ) 보고
- 카테고리별(요건심사 / 상이등급, 신속/보통/심화 트랙) 균형 샘플링

### 4.3 평가표 매핑 (기술성 평가기준 대응)

RFP 평가요소 배점 한도 분석:

| 평가요소 | 배점 | 본 사업 핵심 차별화 |
|---|---|---|
| 데이터 품질 (AI-OCR, VLM, RAG 구축·검증) | 6 | VLM End-to-End + 4계층 평가 |
| 기능 요구사항 | 6 | Hybrid KG-RAG + Agentic Routing |
| 클라우드 서비스 요구사항 | 6 | PPP존 + 범정부 AI 공통기반 활용 |
| 품질 요구사항 | 6 | RAGAS+DeepEval+HHEM 다중 평가 |
| 관리 방법론 | 6 | Drift monitoring 4계층 |
| 사업 이해도 | 7 | VA 사례 + 한국 법률 NLP 경험 |
| 테스트 요구사항 | 5 | Golden dataset + 시나리오 50+ |

**6점 항목 5개를 모두 만점 근접 확보**하는 것이 변별점 (총배점 ~85+).

---

## 5. 휴먼 인 더 루프(HITL) 협업체계 설계 — RFP 추진전략 직접 대응

### 5.1 RFP가 명시한 HITL 전략 — "사람 중심의 AI 협업체계"

본 사업 RFP는 **사업 추진전략 7개 중 하나로 HITL을 명시**하고 있다. 단순 안전장치가 아닌 **사업 정체성의 정의**다.

원문 (RFP 3장 추진계획 / 나. 추진전략):

> **○ (휴먼 인 더 루프) AI는 보조도구로서 초안과 근거를 제시하고, 최종 판단은 보훈심사위원회에서 결정하는 '사람 중심의 AI 협업체계' 정립**
> - 보훈심사 예측 결과와 함께 근거 법령과 유사 사례(의결서·판례 원문 등)를 함께 제시하도록 함으로써 **'의사결정 지원 도구'로서의 위상 명확화**

이 전략은 본 사업의 다른 요구사항에 일관되게 박혀 있다:

- **SFR-006 (심사결과 예측)**: 단순 예측이 아닌 "근거 법령 + 유사 사례 + 등급 분포 분석"을 묶어 제시하는 의사결정 지원
- **SFR-007 (검토서 자동생성)**: "**담당자 및 주심위원 검토 및 수정·편집 등 기능 제공**" — 자동 생성된 검토서를 사람이 검토·수정하는 워크플로를 RFP가 직접 명시
- **SFR-005 (유사사례 추천)**: 유사도 산정방식은 제안사가 제시하되 **발주기관 승인** — 사람이 알고리즘 자체에 개입
- **TER-004**: 50+ 테스트케이스 정답률 90% — 100%가 아닌 점은 **HITL 검증을 전제**로 한 목표 설정

즉 본 사업은 **"AI가 결정한다"가 아니라 "AI가 초안·근거를 만들고, 보훈심사위원회가 결정한다"**가 시스템 위상이다. 이 점을 정확히 설계에 반영한 제안서가 **사업 이해도(배점 7)** 평가에서 결정적 우위를 점한다.

### 5.2 글로벌 규제 동향 — EU AI Act Article 14

본 사업이 향후 한국 「AI 기본법」 시행(2026.1) 및 국제 표준 준수 측면에서 직접 대응해야 할 규제:

- **EU AI Act Article 14 — Human Oversight** (Regulation 2024/1689, **2026년 8월 2일 발효**) — 고위험 AI 시스템에 인간 감독 의무화. 보훈심사처럼 시민의 권리·복지 결정에 관여하는 시스템은 **high-risk 분류 거의 확실**
- **U.S. CFPB** — AI 기반 신용 결정의 explainability 의무화
- **NIST IR 8596** — Human-in-the-loop 검사 권고
- **U.S. AMA(미국의사협회)** — 건강보험 청구 심사에 AI 사용을 빠르게 가속하는 것에 대한 우려와 규제 가드레일 요구 (Health Affairs Forefront, 2024)

본 사업이 EU AI Act Article 14에 부합하는 HITL 아키텍처를 설계해 두면 향후 국제기관 연계·해외 사례 발표 시 큰 자산이 된다. RFP가 명시한 "유관기관이 자유롭게 연계 가능한 개방적 구조" 추진전략과도 연결.

### 5.3 핵심 연구동향 — Learning to Defer (L2D) & Human-AI Complementarity

HITL을 단순 "사람이 마지막에 본다"로 구현하면 자동화 편향(Automation Bias)으로 인해 사람이 거수기로 전락한다. 학계는 **언제·무엇을 사람에게 넘기는가**를 정량적으로 학습하는 Learning-to-Defer 흐름을 발전시켜 왔다.

#### (1) Learning to Defer 핵심 계보

| 연구 | 기여 |
|---|---|
| **Madras et al. (2018)** | L2D 패러다임 최초 제안. AI가 예측할지 사람에게 넘길지 학습 |
| **Mozannar & Sontag (ICML 2020)** | Consistent surrogate loss for cost-sensitive deferral. 의료 영상에서 폐렴 진단 deferral |
| **Verma & Nalisnick (ICML 2022)** | Calibrated L2D with One-vs-All classifiers. **칼리브레이션 보장** |
| **Mao et al. (NeurIPS 2023)** | Two-Stage Learning to Defer with Multiple Experts. 다중 전문가 deferral |
| **DeCCaF (Alves et al., 2024)** | **Cost-sensitive + Workload constraints**. Type I/II 비용 비대칭 + 전문가별 처리 capacity 반영. 본 사업 심사관 워크로드 제약과 직접 매핑 |
| **EA-L2D (Strong et al., 2025)** | Bayesian, unseen experts 일반화 |
| **No Need for Learning to Defer? (arXiv 2509.12573, 2025)** | **Conformal prediction 기반 training-free deferral**. 전문가 annotation 비용이 큰 본 사업에 적합 |

#### (2) 의료·법률·보험 도메인 HITL 운영 사례

본 사업과 가장 유사한 high-stakes 의사결정 시스템 사례:

- **Trustworthy AI for Healthcare: Guided Deferral System** (arXiv 2406.07212, 2024) — LLM이 단순 deferral이 아닌 **deferral 시 가이드(어떤 정보를 더 봐야 하는지)까지 제공**. 본 사업 SFR-006의 "판단 근거 제공"과 정확히 부합
- **DigitalOwl (현 ChartSwap Insights)** — 미국 장애 청구 심사 AI. 의료기록에서 "활동성 일상생활(ADL) 손상 요인" 추출 + 위험 평가 + **빠른 클레임 결정**, 사람이 최종 결정. 본 사업 SFR-001~006와 거의 동일 구조
- **openIMIS AI Claims Adjudication** (Nepal) — **800만 수혜자, 일 8만건 청구** 처리. 세 가지 모드(Rule-based / AI / Manual review) 조합. 본 사업의 신속/보통/심화 트랙 구조와 동형
- **U.S. VA Smart Ratings Recommendation** (2025) — 등급 추천까지 진행하되 **변호사·심사관이 누락 여부 수동 검증** (The War Horse 2025 비판)

#### (3) UX 측면 — 자동화 편향(Automation Bias) 차단

L2D 알고리즘이 좋아도 **UI 디자인이 잘못되면 인간이 자동 도장 찍는 거수기로 전락**한다. 2025년 산업 현장의 핵심 발견:

- **Latency is the killer** — 인간 체크포인트가 항상 처리를 늦춤. UI가 불친절하면 사용자는 서둘러 도장만 찍게 됨 (Millward, Medium 2025)
- **Over-reliance Trap** — 인간 인지 편향에 대응해 **투명성 인디케이터 + 신뢰도 시각화**가 필수. 그렇지 않으면 사람이 자기 전문 지식과 상충해도 기계에 굴복
- **The Algorithmic Automation Problem** (Raghu et al., 2019) — 예측·triage·인간 노력의 균형. 의료 second opinion에서 직접 검증

#### (4) Annotation 효율 — Active Learning

HITL이 만들어내는 **심사위원 수정 이력**은 본 사업의 가장 가치 있는 데이터다. 이를 효율적으로 골든셋·재학습 데이터로 흡수하는 방법:

- **ALANNO** (arXiv 2211.06224) — Active learning annotation system. 라운드 단위 점진적 라벨링, 어노테이터별 작업 분배, 일관성 평가. 본 사업 운영 단계에 그대로 이식 가능
- **Active Annotation** (Nardi et al., JMIR 2021) — 의료 정보 신뢰성 평가에 9명 전문가 + 10,000건 라벨 + 미화 $7,000 효율. 본 사업 도메인 전문가 활용 모델
- **Uncertainty Sampling + Diversity Sampling** (Maxim 2025) — 불확실 사례를 인간에게 우선 라우팅 + 다양성 확보로 blind spot 방지

### 5.4 본 사업 HITL 협업체계 설계

위 연구·사례를 본 사업에 맞춰 4계층 구조로 통합:

#### Layer 1: Confidence 기반 Routing (Learning-to-Defer)

AI 출력에 대한 calibrated confidence를 산출하여 신속/보통/심화 트랙으로 자동 라우팅. 본 사업이 명시한 **"난이도 기반 심사트랙(신속/보통/심화) 자동 배정"** 요구사항과 직접 매핑.

**Confidence 산출 신호 (다중 결합)**:
- LLM 자체 logit 기반 확률 (단, verbalized confidence는 과신 경향 — Galileo 2025)
- Conformal Prediction 기반 정량 보장 (No Need for L2D, 2025)
- 환각 탐지 점수 (HHEM, Lynx)
- RAG 근거 일치도 (RAGAS Faithfulness)
- 유사사례 top-k 일치도 (의결서 유사도 분포)
- OOD 탐지 (학습 데이터 분포 외 안건)

**라우팅 결정 규칙**:
- 신속 트랙: 모든 신호 동의, top-1 유사사례 일치도 0.85+
- 보통 트랙: 일부 신호 불일치 or 중간 유사도
- 심화 트랙: 환각/OOD 의심, 근거 부족, 신규 질병코드 등

**비대칭 비용 반영** — DeCCaF식 cost-sensitive 학습:
- Type I 오류 (인정해야 할 안건을 부인): 보훈대상자 권리 침해 + 행정쟁송 유발 → **고비용**
- Type II 오류 (부인해야 할 안건을 인정): 예산 오용 → 중간 비용
- 비대칭 가중치로 deferral 임계치를 보수적으로 조정

#### Layer 2: 트랙별 HITL 강도

RFP의 신속/보통/심화 트랙 정의를 HITL 강도와 결합:

| 트랙 | HITL 강도 | 검토 인원 | UI 강조점 |
|---|---|---|---|
| **신속** (High Conf) | 담당자 1인 확인 + 사후 샘플링 감사 | 1명 | 핵심 근거 요약, 빠른 확인 인터페이스 |
| **보통** (Mid Conf) | 담당자 + 주심위원 | 2명+ | 출처 위치 명시, 근거 검증 강화 |
| **심화** (Low Conf) | 보훈심사위원회 전체 | 위원회 | AI 결과는 참조 자료로만 한정, 전문가 토의 중심 |

#### Layer 3: UI 설계 원칙 — Automation Bias 차단

HITL의 본질은 UI에 있다. 다음 원칙을 모든 화면에 일관되게 적용:

1. **투명성 인디케이터** — 모든 AI 출력에 confidence 시각화 (0~1, 색상 코딩)
2. **출처 위치 명시 (SFR-007 요구사항)** — 검토서 모든 문장에 원본 의결서·법령 위치 클릭 가능 링크
3. **"AI 의견" 명확 표기** — 사람이 본인 판단을 AI 판단으로 착각하지 않도록 출처 라벨링 (예: "AI 추천", "최근 5년 유사사례 분석")
4. **의도적 마찰 (Productive Friction)** — 심화 트랙에서는 자동 도장 방지를 위해 명시적 근거 입력 요구 (예: "이 판정의 핵심 근거는 무엇입니까?" 텍스트 박스)
5. **대안 시나리오 제시** — "만약 X 증거가 추가된다면 어떻게 달라지는가" 시뮬레이션 (SFR-006의 등급 분포 분석 발전형)
6. **Override 쉬움 + 기록** — 심사위원이 AI 의견을 뒤집는 것을 쉽게, 동시에 모든 override 사유 기록

#### Layer 4: 거버넌스·평가·재학습 연계

HITL 운영 데이터를 평가·재학습 메커니즘과 통합. **이 부분이 6장 운영 단계 지속 최적화와 결정적으로 맞물린다**.

**거버넌스 메트릭**:
- **AI vs 사람 일치율** — 트랙별·CaseType별·심사위원별 분포
- **Override Rate** — 사람이 AI 의견을 뒤집은 비율 (높으면 AI 신뢰도 낮음 또는 사람 자동화 편향 부재)
- **Escalation Rate** — 라우팅으로 인해 심화 트랙으로 올라간 비율 (목표 운영 비용·정확도 트레이드오프)
- **위원별 분산** — 동일 안건에 대한 위원 간 판정 분산 (높으면 일관성 문제)
- **Automation Bias 지표** — 자동 승인 비율·평균 검토 시간·override 후 정확도 비교

**시스템 진화 트리거 연계 (6장 통합) — 모델 학습 없이도 진화**:
- 심사위원의 모든 수정·override 이력이 **자동으로 골든셋 갱신 후보**로 큐잉
- 도메인 전문가 주기 검증 → **Extended 골든셋 회귀 검증 기준선에 편입** (6.4 골든셋 3계층 구조)
- Override 패턴 분석 → **프롬프트 few-shot 갱신 후보** (오답을 negative, 정답을 positive로 — DSPy 자동 컴파일 입력) (6.3 축 2)
- 트랙별 Confidence 분포 → **Adaptive RAG 라우팅 분류기 재학습 데이터** (6.3 축 1)
- 출처 클릭 빈도 → **RAG 메타데이터 우선순위 조정** (6.3 축 1)
- 즉 **사람의 수정이 시스템 진화의 핵심 신호** — Galileo의 "human corrections로 LLM-as-judge 메트릭 개선" 패턴 적용. **LLM 가중치 학습 없이 RAG·프롬프트·라우팅 진화만으로 시스템은 사람의 피드백을 흡수한다**

**EU AI Act Article 14 대응 산출물**:
- Human Oversight 로그 — 모든 AI 의사결정에 대한 사람의 검토·승인·기각 이력
- Bias·일관성 정기 감사 보고서 (분기)
- Override 사유 분석 보고서

### 5.5 RFP 요구사항 직접 매핑

| RFP 요구 | 본 절 대응 메커니즘 |
|---|---|
| **추진전략 5 (휴먼 인 더 루프)** | 본 절 전체. Confidence 기반 Routing + 4계층 강도 + UI 원칙 |
| **SFR-005 (유사사례 추천)**: 발주기관 승인 유사도 산정 | L1 Confidence 신호 중 유사사례 일치도 정의에 발주처 협의 절차 명시 |
| **SFR-006 (예측 + 근거)**: 근거 법령·유사사례 묶음 제시 | UI 원칙 2(출처 위치 명시) + 5(대안 시나리오) |
| **SFR-007 (검토서)**: 담당자·주심위원 검토·수정·편집 | L2 트랙별 HITL 강도 + UI 원칙 4(의도적 마찰) |
| **난이도 기반 심사트랙 자동 배정** | L1 Confidence 기반 Routing 직접 구현 |
| **TER-004 정답률 90%** | 90%가 100%가 아닌 점이 HITL 전제. 나머지 10% 처리를 L2 보통·L3 심화 트랙으로 분산 |

**기술성 평가 점수 영향**:

| 평가 요소 | 배점 | 본 절 기여 |
|---|---|---|
| **사업 이해도** | 7 | HITL 추진전략 정확 이해 + 신속/보통/심화 트랙 연계 |
| **기능 요구사항** | 6 | SFR-005~007의 HITL 측면 구체화 |
| **품질 요구사항** | 6 | Automation Bias 방지 + 거버넌스 메트릭 |
| **관리 방법론** | 6 | L4 거버넌스 계층 + EU AI Act 대응 |

### 5.6 핵심 인용 (Top 10)

#### Learning to Defer / Human-AI Collaboration
1. **Madras et al. (2018)** — Predict Responsibly: Learning to defer for fairer/safer collaboration
2. **Mozannar & Sontag (ICML 2020)** — Consistent Estimators for Learning to Defer (arXiv 2006.01862)
3. **Verma & Nalisnick (ICML 2022)** — Calibrated L2D with OvA Classifiers
4. **Mao et al. (NeurIPS 2023)** — Two-Stage Learning to Defer with Multiple Experts
5. **DeCCaF — Alves et al. (2024)** — Cost-Sensitive L2D with Workload Constraints (arXiv 2403.06906). 본 사업과 가장 직접 부합
6. **No Need for L2D? (arXiv 2509.12573, 2025)** — Training-free conformal deferral

#### 의료·법률·보험 HITL 실증
7. **Trustworthy AI for Healthcare: Guided Deferral** (arXiv 2406.07212, 2024)
8. **DigitalOwl / ChartSwap Insights** — 미국 장애 청구 AI 운영 사례
9. **VA Smart Ratings Recommendation** (2025) + The War Horse 비판 — 좋은 점과 한계 모두

#### 규제·UX
10. **EU AI Act Article 14** (Regulation 2024/1689, **2026.8.2 발효**) — 고위험 AI 인간 감독 의무화. NIST IR 8596, U.S. CFPB explainability

---

## 6. 운영 단계 지속 최적화 — RAG 인덱스 + 프롬프트 진화 + Drift 모니터링 3축

### 6.1 RFP 문언 충실 — "도메인 특화 학습"의 실제 의미

본 사업 RFP는 추진전략 4번에서 다음과 같이 명시한다:

> **(도메인 특화 학습) 범정부 AI 공통기반 LLM을 활용하되, RAG DB 구축 및 프롬프트 최적화를 통해 보훈 도메인(보훈법령, 의학, 군사 용어 등) 정확도 확보**
> - 법률 및 의학 전문용어가 혼재된 전문적인 텍스트를 생성형AI모델(LLM)이 정확히 이해하고 추론하도록 **반복적인 최적화(Optimization) 수행**

핵심 함의를 정확히 짚어두는 것이 평가 단계에서 결정적이다:

- **"활용하되"** — LLM은 이미 학습된 범정부 AI 공통기반(NIA 제공)을 그대로 활용. 제안사가 LLM 가중치에 손대지 않음
- **"RAG DB 구축 및 프롬프트 최적화"** — 명시된 도메인 적응 수단은 두 가지뿐. **파인튜닝(LoRA, QLoRA, Full FT 등)은 RFP에 등장하지 않음**
- **"반복적인 최적화(Optimization)"** — Optimization은 학습(training)과 구별되는 개념. RAG 파이프라인 튜닝·프롬프트 진화·검색 파라미터 조정을 가리킴

또한 SFR-002 "AI 학습 데이터셋 구축"도 RFP 내부에서 자체 정의하고 있다:

> "AI 모델 서빙에 활용할 수 있는 **데이터셋(RAG)** 구축 기능 제공" (SFR-002 본문)

즉 본 RFP에서 **"학습 데이터셋"이라는 단어는 ML 학계의 supervised training dataset이 아닌, RAG 코퍼스(벡터 임베딩 + 메타데이터 + 청킹)를 의미**한다. 이는 공공부문 RFP의 일반적 용어 사용 패턴이며, 평가위원이 "학습 데이터셋"이라는 단어를 보고 파인튜닝을 기대한다고 가정하면 곤란하다.

따라서 본 장은 **모델 가중치 학습 없이, RAG 인덱스·프롬프트·라우팅의 진화만으로 운영 단계 지속 최적화를 달성하는 메커니즘**을 설계한다.

### 6.2 운영 관점의 보훈심사 시스템 — "사업 종료는 운영 시작"

의결서 47만건은 사업 종료 시점의 스냅샷일 뿐이며, 운영 중에는 다음이 지속적으로 추가·변경된다:

- **매월 신규 의결서** — 운영 1년 차에만 약 2만건 추가 예상
- **법령·시행령 개정** — 보훈보상관계법령은 연 평균 5–10건 개정
- **신규 질병 코드** — ICD-11 도입(2022) 등 의학 분류 체계 변화
- **새로운 직무·환경성 요인** — 외상후스트레스장애, 화학물질 노출 등 신규 인정 사례

이러한 변화에 RAG 시스템이 자동 대응하지 못하면 **시간이 지날수록 부정확해진다**. 학계 실증:

- **RAG or Learning? Knowledge Drift Benchmark** (arXiv 2604.05096, 2026) — 시간순 실세계 지식 변화 하에서 vanilla RAG도 temporal inconsistency에 취약. RAG는 인덱스를 갱신해야만 진화함
- **Magesh et al. (2024)** — 상용 법률 RAG도 17% 환각률. 운영 중 지속 검증·갱신 없으면 악화

**5장 HITL과의 결합 — 핵심**: 운영 단계 지속 최적화의 가장 가치 있는 학습 신호는 **5장 HITL 협업체계가 만들어내는 심사위원 수정·override 이력**이다. 본 장의 3축은 5장 HITL이 생성하는 신호를 자동으로 흡수해 시스템을 진화시키는 메커니즘이다. **모델 가중치 학습 없이도 사람의 수정이 시스템을 진화시킨다.**

### 6.3 핵심 스토리라인 — 3축 + LangGraph 지휘

> **"RAG 인덱스는 Drift 모니터링이 갱신 시점을 결정하고, 프롬프트는 골든셋 기반으로 진화하며, 라우팅은 워크플로별로 분리되고, 전체는 LangGraph가 지휘한다"**

#### 축 1: RAG 인덱스 진화 (DAR-010 "RAG Optimization" 직접 매핑)

RAG 시스템 자체를 지속 갱신·최적화하는 축. 본 사업 DAR-010이 명시한 "RAG Optimization을 포함하여 검증 후 개선 방안 제시" 요구에 정면 대응.

**Embedding Drift 방지 4원칙** (의성군 작업에서 정립, 본 사업에 그대로 유효):

1. **파이프라인 버전 고정** — 임베딩 모델·전처리·청킹을 하나의 버전으로 관리
2. **부분 재임베딩 금지** — 임베딩 모델 변경 시 전체 코퍼스 재임베딩 (부분 재임베딩은 drift의 #1 원인)
3. **벡터 프로비넌스 저장** — 모든 벡터에 (모델 버전, 전처리 해시, 타임스탬프) 메타데이터
4. **증분 업데이트는 동일 파이프라인 버전으로** — 매월 신규 의결서 추가 시 반드시 기존 파이프라인 버전 적용

**최신 RAG 진화 연구·운영 패턴**:

- **Self-RAG** (Asai et al., ICLR 2024 / arXiv 2310.11511) — 검색 필요성 자체를 LLM이 reflection token으로 판단. 검색→평가→재검색 또는 생성 가지 결정
- **CRAG (Corrective RAG)** (Yan et al., ICLR 2025) — **경량 retrieval evaluator로 검색 품질 평가**. {Correct / Incorrect / Ambiguous} 3단계 confidence로 후속 행동 결정. Self-RAG와 달리 LLM instruction-tuning 불필요 — 본 사업의 범정부 AI 공통기반 활용 가정에 더 적합
- **Adaptive-RAG** (Jeong et al., NAACL 2024) — T5 분류기로 질의 복잡도를 (no-retrieve / single-step / iterative) 3분류. 비싼 baseline 대비 동등 성능 + 비용 절감
- **RAGRouter-Bench** (Wang et al., arXiv 2602.00296, 2026) — RAG 라우팅 전용 벤치마크. 5개 RAG 패러다임(LLM-Only / Naive RAG / GraphRAG / Hybrid / Iterative) 비교, **단일 패러다임이 보편적 우위를 갖지 않음** 실증. 질의-코퍼스 적합성 기반 라우팅 필요
- **HyDE, Multi-query, Step-back, MQRF-RAG** — 쿼리 재작성 전략. HotpotQA에서 HyDE 대비 MQRF-RAG가 7% 향상
- **Versioned Vector DB** (LiveVectorLake, arXiv 2601.05270) — Dual-tier 아키텍처: 실시간 의미 검색 + 완전한 버전 이력. SFR-007 "검토서 자동생성 양식·규칙의 이력관리"와 직접 매핑
- **Index Versioning** (Safjan 2026-02) — 4단계 버전 관리 (Level 0 메타데이터 태그 → Level 4 완전 재현). 시민 권리 직결 시스템은 **Level 3~4 권장**
- **Self-healing RAG with Auto Drift Rollback** — Golden Questions DB + 임베딩 drift 분석 + 자동 promote/rollback

**본 사업 적용 운영 패턴**:

- 매월 신규 의결서를 staging 인덱스에 적재 → 골든셋 회귀 평가 통과 → alias 전환 (Qdrant collection alias 또는 Elasticsearch alias 기반 zero-downtime)
- 평가 회귀 발생 시 alias를 이전 버전으로 즉시 전환 (auto-rollback)
- CaseType별 메타데이터 필터 우선순위 동적 조정 (HITL override 패턴 반영)

#### 축 2: 프롬프트 진화 (RFP "프롬프트 최적화" 직접 매핑)

수동 프롬프트 엔지니어링은 확장 불가능하고 재현성이 낮다. 최신 연구는 **프롬프트를 코드처럼 다루는** 자동 최적화로 이동하고 있다.

**핵심 연구·도구**:

- **DSPy** (Khattab et al., Stanford) — 프로그래밍 방식 LLM 시스템 구축 프레임워크. Signature(선언적 입출력 정의) + Optimizer(자동 프롬프트·few-shot 탐색)로 구성. **"프롬프트 엔지니어링이 아니라 ML 프레임워크"** — 학습 데이터로 프롬프트를 자동 컴파일
- **DSPy 5개 사용사례 실증** (Lemos et al., arXiv 2507.03620, 2025) — guardrail, 환각 탐지, 코드 생성, 라우팅 에이전트, 프롬프트 평가. **프롬프트 평가 task에서 정확도 46.2% → 64.0%** (수동 대비 자동 최적화의 효과)
- **Promptomatix** (Salesforce AI Research, arXiv 2507.14241, 2025) — 자연어 task 설명을 고품질 프롬프트로 자동 변환. Meta-prompt 최적화 + DSPy 컴파일러 모듈, 합성 데이터 자동 생성 + 비용 인식 목적함수
- **TextGrad** — 자연어 피드백을 gradient처럼 사용하여 프롬프트·LLM 파이프라인 자동 개선
- **OPRO (Optimization by PROmpting)** — LLM을 메타 옵티마이저로 사용해 프롬프트 자체를 진화시킴

**본 사업 적용 운영 패턴**:

- **CaseType별 시스템 프롬프트 분리** — 요건심사 / 상이등급 / 법령매칭 / 판례검색 / 검토서 5종. LangGraph가 워크플로별로 라우팅 후 적합한 시스템 프롬프트·few-shot 묶음 적용
- **Few-shot 예시 풀 동적 갱신** — 골든셋 + HITL 검증 의결서 풀에서 task별 유사도 top-k 동적 선택 (RAG 패턴을 prompt context에도 적용)
- **DSPy 기반 자동 최적화 파이프라인** — 골든셋을 ground truth로 사용, 정기적(분기) 프롬프트·few-shot 재컴파일
- **프롬프트 버전 관리** — 모든 프롬프트에 (버전, 생성일, 평가 지표) 메타데이터. Git 기반 변경 이력 + 회귀 비교
- **A/B 테스트 게이트** — 새 프롬프트는 staging에서 골든셋·HITL feedback 평가 통과 후 production 승격

#### 축 3: 4계층 Drift 모니터링 — RAG/프롬프트 갱신 Trigger Decider

CF가 아닌 RAG/프롬프트 갱신 시점을 정확히 판단하는 모니터링 체계. 의성군 작업에서 정립된 4계층 구조가 그대로 유효하되, Red 단계 대응만 RAG·프롬프트 갱신으로 재정의.

**4계층 Drift 모니터링**:

| 계층 | 모니터링 대상 | 도구 | 주기 |
|---|---|---|---|
| **L1 Infrastructure** | GPU 사용률·메모리·응답 지연·동시접속 | Prometheus + Grafana | 실시간 |
| **L2 Retrieval Quality** | Recall@k, MRR, NN Overlap, 임베딩 안정성, 코퍼스 갱신률 | Evidently + Golden Query | 주간 |
| **L3 Response Quality** | Faithfulness, Answer Relevancy, BERTScore, 환각 발생률 | RAGAS + DeepEval | 일간 샘플 |
| **L4 Business Quality** | **쟁송 제기율 변화 · 심사 일관성(동일 사례 다른 판정률) · 심사 적체 트렌드 · 민원 만족도 · "근거 없음" 응답률** | RDB 통계 + 대시보드 | 일/주/월/분기 |

L4 메트릭은 **보훈심사 사업 KPI에 직결**되도록 재정의했다. 특히 **"심사 일관성"**은 본 사업 핵심 KPI인 "행정쟁송 제기건수 30% 이상 감소" 달성 여부를 직접 측정.

**4단계 에스컬레이션 — RAG/프롬프트 갱신은 Red 단계에서만**:

| 수준 | 조건 | 대응 (RFP 충실, 모델 학습 없음) |
|---|---|---|
| 🟢 **Green** | 정상 범위 | 대시보드 기록 |
| 🟡 **Yellow** | L3/L4 10~15% 하락 | **프롬프트 파라미터 튜닝** (temperature, top-p, few-shot 개수), top-k 조정 |
| 🟠 **Orange** | L2 하락 (Recall@k < 0.85), Embedding drift 탐지 | **RAG 인덱스 부분 갱신** (신규 의결서 일괄 임베딩, reranker 가중치 재조정, 메타데이터 필터 갱신) |
| 🔴 **Red** | L2+L3 동시 하락, 심사 일관성 25%+ 하락, 쟁송율 급증 | **종합 갱신**: ① 전체 RAG 인덱스 재구축(필요시 청킹·임베딩 모델 변경 검토 후 발주처 승인), ② DSPy 자동 프롬프트 재컴파일, ③ Adaptive RAG 라우팅 분류기 재학습, ④ 골든셋 자동 회귀 검증 → 통과시 배포, 실패시 rollback |

### 6.4 Golden Dataset 회귀 검증 — 모든 변경의 게이트

본 사업 TER-004가 명시한 **50+ 테스트케이스 골든 데이터셋**을 단순 평가용이 아닌 **모든 RAG·프롬프트 변경의 회귀 검증 기준선**으로 위치 재정의.

**3계층 골든셋 구조**:

| 계층 | 구성 | 활용 |
|---|---|---|
| Core | TER-004 50+ 테스트케이스 (도메인 전문가 검증) | 매 변경 100% 평가 — 회귀 시 자동 차단 |
| Extended | 사업기간·운영기간 중 도메인 전문가 검증 의결서 + HITL 수정 케이스 ~500건+ | 분기별 종합 평가 + 라우팅 분류기 학습 데이터 |
| Adversarial | 의도적 어려운/모호한 케이스 (edge case) | 시스템 견고성 측정 |

**평가 메트릭 — RFP 명시 + α**:

| 영역 | RFP 명시 | 추가 차별화 |
|---|---|---|
| Retrieval | F1, Precision, Recall | MRR, nDCG, **Context Recall, Context Precision (RAGAS)** |
| Generation | BLEU, ROUGE | **Faithfulness, Answer Relevancy (RAGAS), HHEM/Lynx 환각** |
| End-to-end | 정답률 (TER-004 90% 목표) | **AI vs 사람 일치율, Override Rate (5장 HITL 연동)** |

**자동화된 변경 게이트** (Promote / Rollback):

1. 변경(RAG 인덱스 / 프롬프트 / 라우팅) 발생 → staging 환경 자동 배포
2. 골든셋 자동 회귀 평가 (RAGAS + RFP 메트릭)
3. 기준선 대비 회귀 임계치 초과 시 **자동 차단 + 사유 보고**
4. 통과 시 **canary 배포 (5% → 25% → 100%)**
5. 각 단계에서 L1~L4 모니터링 정상 시 다음 단계
6. 회귀 발생 시 즉시 alias rollback

### 6.5 LangGraph 오케스트레이터 — 4축의 화룡점정

RAG·프롬프트·라우팅·모니터링을 단일 프레임워크로 지휘.

**보훈심사 워크플로**:

```
질의 진입
   ↓
[Intent Classifier] — 어떤 워크플로 단계인가?
   ↓
[Adaptive RAG Router] — 질의 복잡도 + 코퍼스 적합성 (RAGRouter-Bench식)
   ├─ Factual (단순 사실) → 빠른 RAG (top-k 작게, 단일 검색)
   ├─ Reasoning (multi-hop) → CRAG / Iterative RAG
   └─ Summarization → GraphRAG (KG 활용)
   ↓
[Policy Router] — 어떤 프롬프트 + KG/VDB 조합?
   ├─ 요건심사 모드 → 요건 프롬프트 + 법령 KG + 판례 VDB
   ├─ 상이등급 모드 → 등급 프롬프트 + 의학 KG + 의무기록 VDB
   ├─ 유사사례 모드 → 판례 프롬프트 + 의결서 VDB (hybrid retrieval)
   ├─ 검토서 모드   → 검토서 프롬프트 + 표준 양식 템플릿
   └─ 대국민 모드   → 안전 가드레일 강화 + 일반 RAG
   ↓
[검색·추론·생성] → [CRAG 평가] → [Faithfulness 검증] → [출처 인용 첨부]
   ↓
[5장 HITL 협업체계 진입 — Confidence 기반 트랙 라우팅]
   ↓
[응답 + 운영 로그(L1~L4 지표 기록)]
```

**LangGraph 강점**:
- **상태 기반 실행** — 다단계 추론 흐름(요건심사 → 유사사례 → 검토서) 자연스럽게 표현
- **Tool Registry** — 외부 시스템 연계(통합보훈정보시스템, 의무기록 시스템) Tool로 추상화
- **체크포인트·재시도** — 장시간 처리되는 검토서 생성도 안정적 처리
- **관찰 가능성** — LangFuse 연동으로 운영 단계 디버깅·평가 일원화

### 6.6 5장 HITL과의 폐쇄 루프 — "사람의 수정이 시스템을 진화시킨다"

5장에서 설계한 HITL 협업체계가 본 장의 3축과 폐쇄 루프(Closed Loop)를 형성한다. **모델 가중치 학습 없이도 사람의 수정이 시스템 진화의 핵심 신호로 작동**한다.

| HITL 신호 (5장) | 6장 흡수 메커니즘 |
|---|---|
| 심사위원 수정 의결서 | **골든셋 Extended 자동 확장 후보** → 도메인 전문가 검증 후 회귀 검증 기준선에 편입 |
| Override 패턴 (AI 의견 뒤집힘) | **프롬프트 few-shot 갱신 후보** — 오답 케이스를 negative example로, 정답을 positive로 |
| 트랙별 Confidence 분포 | **Adaptive RAG 라우팅 분류기 재학습 데이터** — 신속/보통/심화 라우팅 정확도 개선 |
| 출처 위치 클릭 빈도 | **RAG 메타데이터 우선순위 조정** — 자주 참조되는 법령·판례에 가중치 부여 |
| 검토 시간 통계 | **L4 Business Quality** 메트릭 — 평균 검토 시간 단축 KPI 달성 측정 |

이 루프가 본 사업의 핵심 차별화 메시지를 완성한다: **모델은 학습하지 않지만, 시스템은 진화한다.**

### 6.7 운영 단계 메트릭 정리

본 사업 RFP가 명시한 정적 메트릭(F1·Precision·Recall·BLEU·ROUGE)을 운영 단계로 확장:

| 메트릭 | 정의 | 본 사업 적용 |
|---|---|---|
| **RAG Recall@k 변동** | 시간에 따른 검색 정확도 변화 | 매주 골든셋 평가, Yellow/Orange 트리거 |
| **Faithfulness Drift** | 응답의 컨텍스트 충실도 변화 | RAGAS 일일 샘플, Red 트리거 |
| **Override Rate** | 사람이 AI 의견을 뒤집은 비율 | 5장 HITL 연동, 트랙별 추적 |
| **AI vs 사람 일치율** | 트랙·CaseType별 분포 | 자동화 편향 vs 정확성 동시 측정 |
| **Escalation Rate** | 상위 트랙으로 라우팅된 비율 | 운영 비용·정확도 트레이드오프 모니터링 |
| **심사 일관성 지표** | 동일 사례의 다른 판정률 | 핵심 KPI "쟁송 30% 감소"와 직결 |
| **Prompt Regression** | 프롬프트 변경 후 골든셋 성능 변화 | 자동 게이트 (Promote/Rollback) |

### 6.8 RFP 요구사항 직접 매핑

| RFP 요구 | 본 절 대응 메커니즘 |
|---|---|
| **추진전략 4 (도메인 특화 학습)** | RAG DB 구축 + 프롬프트 최적화 정면 대응 (LLM 가중치 학습 없이) |
| **SFR-002 (AI 학습 데이터셋 = RAG 구축)** | 축 1 RAG 인덱스 진화 + 골든셋 회귀 검증 |
| **DAR-010 (RAG Optimization + 검증 후 개선)** | Self-RAG/CRAG/Adaptive-RAG + Embedding Drift 4원칙 + Versioned Vector DB |
| **DAR-007 (데이터 관리체계 — 표준 준수율 등)** | 4단계 에스컬레이션 + 자동 회귀 게이트 |
| **QUR-001 (품질 지표별 목표 수준)** | 운영 단계 메트릭 7종 정량화 |
| **QUR-005 (시스템 장애 최소화)** | Self-healing RAG의 auto-rollback |
| **QUR-006 (시스템 변경사항 이력관리)** | Versioned Vector DB + Prompt Versioning Level 3~4 |
| **SFR-007 (검토서 양식·규칙 이력관리)** | LiveVectorLake식 dual-tier 아키텍처 |

**기술성 평가 점수 영향**:

| 평가 요소 | 배점 | 본 절 기여 |
|---|---|---|
| 데이터 품질 요구사항 | 6 | RAG Optimization + Embedding Drift 4원칙 + Versioned Vector DB |
| 품질 요구사항 | 6 | DSPy 자동 최적화 + RAGAS + 자동 회귀 게이트 |
| 관리 방법론 | 6 | 4계층 Drift + 🟢🟡🟠🔴 + Auto-rollback |
| 클라우드 서비스 요구사항 | 6 | PPP존 RAG 인덱스 운영 + Self-healing |

### 6.9 핵심 인용 (Top 12)

#### RAG 진화 (자기 평가·교정)
1. **Self-RAG** — Asai et al., ICLR 2024 (arXiv 2310.11511). Reflection token 기반 검색·생성 자기 평가
2. **CRAG** — Yan et al., ICLR 2025. 경량 retrieval evaluator + Correct/Incorrect/Ambiguous 3분기. **본 사업 적합도 최상**
3. **Adaptive-RAG** — Jeong et al., NAACL 2024. 질의 복잡도 분류 기반 라우팅
4. **RAGRouter-Bench** — Wang et al., arXiv 2602.00296 (2026). 5개 RAG 패러다임 비교 벤치마크

#### 프롬프트 자동 최적화
5. **DSPy** — Khattab et al., Stanford. Signature + Optimizer 프레임워크
6. **DSPy 5개 사용사례** — Lemos et al., arXiv 2507.03620 (2025). 프롬프트 평가 46.2% → 64.0%
7. **Promptomatix** — Salesforce, arXiv 2507.14241 (2025). 자연어→프롬프트 자동 변환
8. **TextGrad / OPRO** — Stanford / DeepMind. Gradient-style / meta-optimization

#### Drift / Versioning / Self-healing
9. **LiveVectorLake** — arXiv 2601.05270. Dual-tier versioned vector DB
10. **Index Versioning** — Safjan 2026-02. 4-level versioning model
11. **RAG or Learning? Knowledge Drift Benchmark** — arXiv 2604.05096. RAG와 fine-tuning 모두 한계 실증

#### 쿼리 진화
12. **MQRF-RAG** (multi-query rewriting) — ACM 2025. HotpotQA에서 HyDE 대비 7% 향상

---

## 7. 핵심 인용 논문·자료 정리

### 7.1 직접 적용 가능 논문 (Top 12)

1. **Hindi et al. (2025)** — Enhancing the Precision and Interpretability of RAG in Legal Technology: A Survey. IEEE Access. → 법률 RAG 메트릭 체계
2. **Hou et al. (2024)** — CLERC: A Dataset for Legal Case Retrieval and RAG. arXiv 2406.17186 → 법률 RAG 데이터셋
3. **Pipitone & Alami (2024)** — LegalBench-RAG. arXiv 2408.10343 → 법률 RAG 벤치마크
4. **Zheng et al. (2025)** — A Reasoning-Focused Legal Retrieval Benchmark. CS&Law '25 → BM25/Dense 한계 실증
5. **Domain-Partitioned Hybrid RAG** (arXiv 2602.23371, 2025) → 70% vs 37.5% pass rate 근거
6. **HyPA-RAG** (NAACL 2025 Industry) → 적응형 RAG
7. **UQLegalAI@COLIEE 2025** (arXiv 2505.20743) → LLM + GNN 법률 검색
8. **ReaKase-8B** (arXiv 2510.26178, 2025) → 추론·트리플릿 기반 사례 검색
9. **PredEx** (arXiv 2406.04136) → LJP + 설명 데이터셋
10. **LexFaith-HierBERT** (Nature Scientific Reports 2026) → faithfulness-aware LJP
11. **Magesh et al. (2024)** — Hallucination-Free? 상용 법률 RAG 17% 환각 → 환각 대응 근거
12. **Es, James et al. (EACL 2024)** — RAGAS → 평가 프레임워크 표준

### 7.2 운영 사례 (Top 5)

1. **VA Booz Allen Hamilton IDP** — Amazon Textract, 1,300페이지/케이스 처리. <https://meritalk.com/articles/how-the-va-leveraged-ai-to-accelerate-claims-processing/>
2. **VA AICES + Smart Ratings Recommendation** (2025) — 등급 추천 단계 진입
3. **Naver HyperCLOVA X + Continental Aju** — 한국어 법률 RAG 운영 사례
4. **Fullerton Health** — 95% 분류, 87% 추출 F1, 2초 미만 지연 (arXiv 2601.01897)
5. **Microsoft GraphRAG** — community 계층 사전요약, 20–70% 향상

### 7.3 평가 도구 / 프레임워크

- **RAGAS** — github.com/explodinggradients/ragas
- **DeepEval** — github.com/confident-ai/deepeval
- **Lynx** — HaluBench 환각 탐지
- **HHEM (Vectara)** — Hughes Hallucination Evaluation Model
- **LegalBench** — 162개 법률 추론 태스크 (NeurIPS 2024)
- **COLIEE** — 매년 갱신 (Task 1~5)
- **Microsoft Presidio** — PII/PHI 탐지·마스킹

---

## 8. 종합 결론

본 사업의 5대 핵심 기능은 모두 2024–2025년 학계·산업계에서 **표준화된 baseline이 존재**하는 영역이다. 단순히 RAG/LLM을 적용하는 것으로는 차별화가 어렵다.

### 8.1 경쟁우위 확보 5대 축

1. **아키텍처 차별화** (2장·4장) — Hybrid KG-RAG + Agentic Routing. 단순 RAG 대비 70% vs 37.5% pass rate (Domain-Partitioned Hybrid RAG 2025), GraphRAG 20–70% 향상, Agentic RAG HotpotQA 94.5% 등 정량 근거. 본인의 한국 세법 KG-RAG 시스템 자산이 동형 매핑으로 직접 이식
2. **품질 검증 차별화** (3장) — 4계층 평가 (VLM → Retrieval → Generation → Domain). RFP 명시 메트릭(F1·Precision·Recall·BLEU·ROUGE)을 RAGAS Faithfulness/Answer Relevancy + HHEM/Lynx 환각 탐지로 보강. **Magesh et al. 17% 환각** 실태에 대한 명시적 대응
3. **운영 사례 차별화** (2장 SFR-001~003) — VA Booz Allen Hamilton IDP 사례 직접 인용. 케이스당 1,300페이지 처리, 우편처리 10일→0.5일 단축 등 보훈 도메인 운영 베이스라인 제시. HITL 메커니즘 + 누락 탐지 메트릭으로 보완
4. **HITL 협업체계 차별화** (5장) — **RFP가 추진전략으로 명시한 "사람 중심 AI 협업"** 을 정면 구현. Learning-to-Defer 기반 Confidence Routing → 신속/보통/심화 트랙별 HITL 강도 차별화 → Automation Bias 차단 UI 원칙 → 거버넌스 연계. **EU AI Act Article 14 (2026.8.2 발효)** 선제 대응
5. **운영 단계 지속 최적화 차별화** (6장) — **RFP 문언 충실**한 "RAG 인덱스 + 프롬프트 진화 + Drift 모니터링" 3축 + LangGraph 지휘 통합 아키텍처. **모델 가중치 학습 없이 (RFP가 명시한 RAG·프롬프트 최적화 수단만으로)** SFR-002·DAR-010 "지속적 최적화" 요구를 구체적 메커니즘으로 구현. Self-RAG / CRAG / Adaptive-RAG / DSPy / Versioned Vector DB / Self-healing 등 최신 연구 직접 적용. **5장 HITL의 심사위원 수정 이력이 골든셋·프롬프트·라우팅 진화 신호로 자동 흐름** (Closed Loop)

### 8.2 핵심 차별화 메시지

> **"보훈심사 시스템은 사람을 대체하는 AI가 아니라 사람을 보조하는 AI다. AI가 초안·근거를 제시하고 보훈심사위원회가 결정하며, 그 결정 이력이 다시 시스템을 진화시킨다. 모델은 학습하지 않지만, 시스템은 진화한다 — RAG 인덱스는 Drift 모니터링이 갱신 시점을 결정하고, 프롬프트는 골든셋이 진화시키며, 라우팅은 워크플로별로 분리되고, 전체는 LangGraph가 지휘한다."**

이는 단순 구축 사업자와의 변별점이며, **RFP 추진전략(휴먼 인 더 루프, 도메인 특화 학습 = RAG/프롬프트 최적화, 지속적 AI 최적화)을 문언 그대로 정면 흡수**한 결과. 본인의 의성군 작업에서 검증된 운영 단계 스토리라인을 보훈심사 도메인 특성(시민 권리 직결, high-stakes) 및 범정부 AI 공통기반 활용 가정(LLM 가중치 동결)에 맞춰 재구성한 자산.

### 8.3 기존 작업 자산 활용 매핑

| 기존 자산 | 보훈심사 적용 |
|---|---|
| 한국 세법 KG-RAG 시스템 (Hybrid Neo4j + Qdrant + LLM) | 보훈심사 GDB·VDB 아키텍처 그대로 이식. 5 CaseType × 다층 법령 구조 동형 |
| 의성군 3축 (Drift + LangGraph) | 본 보고서 6장으로 직접 반영. Multi-LoRA는 RFP 미명시로 제외, 대신 Multi-Prompt / DSPy 자동 최적화로 변형 |
| 4계층 Drift monitoring 설계 (L1~L4) | L4 메트릭을 보훈심사 KPI(쟁송율·일관성·적체)로 재정의하여 그대로 사용 |
| Embedding Drift 4원칙 | DAR-010 직접 매핑. 매월 신규 의결서 추가에 결정적 |
| OSS LLM(`gpt-oss-120b`) 친화 few-shot 패턴 27개 | 범정부 AI 공통기반 환경에서 프롬프트 few-shot 풀에 직접 활용 |
| 법제처·수원시 제안 경험 (법률 도메인 RAG, 평가 체계) | 평가표 매핑 노하우 + 법률 도메인 신뢰성 |

### 8.4 평가 부문 점수 영향 종합

배점 한도 30점 중 본 보고서 전 영역이 직간접 기여하는 부분이 **6점 항목 5개 (총 30점) + 사업 이해도 7점**:

| 평가 요소 | 배점 | 기여 영역 |
|---|---|---|
| 데이터 품질 요구사항 | 6 | 2장 VLM End-to-End + 6장 RAG 인덱스 진화 + Embedding Drift 4원칙 |
| 기능 요구사항 | 6 | 2장 Hybrid KG-RAG + 5장 HITL UI 원칙 + 6장 LangGraph 오케스트레이터 + Adaptive RAG |
| 클라우드 서비스 요구사항 | 6 | 6장 PPP존 RAG 운영 + Versioned Vector DB + Self-healing |
| 품질 요구사항 | 6 | 3장 RAGAS/DeepEval + 5장 Automation Bias 차단 + 6장 자동 회귀 게이트 (DSPy) |
| 관리 방법론 | 6 | 5장 거버넌스 계층(EU AI Act 대응) + 6장 4계층 Drift + 🟢🟡🟠🔴 |
| **사업 이해도** | **7** | **5장 HITL 추진전략 정면 흡수 + 6장 "도메인 특화 학습 = RAG/프롬프트 최적화" 정확 해석** + 2장 VA 사례 |
| 테스트 요구사항 | 5 | 3장 Golden dataset + 시나리오 + 6장 Continual Learning 평가 |
