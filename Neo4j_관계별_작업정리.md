# Neo4j 관계별 작업 정리 (심판·판례·예규 · 순수 Cypher)

조세심판 KG-RAG 그래프의 관계 작업을 **대상 쌍별로** 재편.
전부 Cypher(+APOC/GDS) — PG 의존 없음.

**제약** — Neo4j CE + APOC 가용 · GDS `concurrency ≤ 4` · 쓰기는
`apoc.periodic.iterate` 배치 · 실행 전 대상 count 확인.

---

## 실측 기준선

**⚠ 두 빌드를 구분할 것** — 아래 지표 중 다수는 **구 빌드** 측정치이며,
신규 빌드에서 재측정이 필요하다(관계 총량이 달라졌으므로 비율도 변동).

### 신규 빌드 (재구축본, 현행)

| 항목 | 값 | 비고 |
|---|---|---|
| ADOPTS_INTENT(참조결정) | **50,005 → 51,139** | 원본 50,005 + §2-1 보강 1,134(`source='followed_inverse'`) |
| EXAMINES_DECISION(따른결정) | **40,831** (citation 37,634 / merged 3,197) | 구 빌드 35,315 → 증가. merged가 B-1 예측(3,197)과 정확히 일치 |
| `reference_date` 커버리지 | **사실상 완전** | day_gap 산출이 총량과 거의 일치 = 결정일 결측 미미 (구 빌드 reg_date 전량 null과 대조) |
| day_gap 구간별 상호비율 | **동일자 49.7% / 1주내 47.2%** / 1개월내 1.7% / 1년내 1.1% / 1년초과 1.3% | **임계 7일 확정** (self-loop 제거 후 재측정, §1-2 B-1) |
| 병합 후보(≤7일) | **3,197건** (동일자 2,947 + 1주내 250) | self-loop 제거 후 확정 |
| RELATED_CASE | **2,405 무방향 엣지** | §2-2 완료. 세목 순도 100%, 최대 차수 26. 전국 접수분 동일쟁점 일괄처리 구조 |
| CaseName 분류 | substantive 262,805 / procedural 6,134 / case_title 341 | 절차·사건명 격리 완료 |
| Issue(축 A) | 문자열 8,533(교차 741) + **키워드 20개 전량 3유형 관통** | §2-3 축 A 확정 — 다리는 문장이 아니라 쟁점 어휘 |
| 편측 참조결정 | **10,728** (21.5%) | ADOPTS에만 있는 고유 정보 |
| 편측 따른결정 | **1,545** (3.8%) | EXAMINES에만 있는 고유 정보 → **EXAMINES는 ADOPTS 역방향에 96.2% 포함** |
| self-loop | **38건** (EXAMINES 기준, 제거 완료) | ETL 자기참조 버그. 영향 미미(동일자 2,985→2,947), 판정 불변. ETL 필터 추가 권장 |
| 연도 오타 노드 | 실측: `국심1192서0547`(1192), `국심1955서3399`(결정 1996, 41년차) | 원천 기재 오타가 ETL 통과. 접수·결정연도 대조로 탐지 → **§0-B-1** |
| 사건번호 자릿수 변이 | 실측 예: `국심1991서00333` | 5자리 표기 — 동일 사건 분리 적재 가능성 → **§0-B-2** |
| 같은방향 중복(B-3) | 2,984 | 동일자 2,328(78%) · 1개월초과 530(17.8%) — 후자는 기재 오류 후보 |
| ⚠ 비대칭 방향 전환 | ADOPTS(5만) − EXAMINES(4만) ≈ 1만 | 구 빌드는 거의 대칭(29,420 vs 29,226) → 신규는 **참조결정이 주 기록, 따른결정은 그 부분 역인덱스** 구조 |

### 구 빌드 (초기 분석 근거 — 신규 빌드 재측정 대상)

| 항목 | 값 | 비고 |
|---|---|---|
| 심판 문서 | 78,005 | |
| EXAMINES_DECISION | 29,226 문서 보유 / 35,315 엣지 | 37.5% 커버리지 |
| ADOPTS_INTENT | 29,420 문서 보유 | |
| 역방향 대응 | 25,716 (72.8%) | 두 관계가 같은 인용의 양면 — **재측정 필요** |
| 같은방향 중복 | 2,416 (6.8%) | 시간상 모순, 병합 가설 검증 대상(§1-2 B-3) — **재측정 필요** |
| 방향(사건번호 연도 대조) | 과거→미래 80.4% / 동년 18.5% / 미래→과거 1.1% | 접수연도 기준이라 과대추정 — day_gap으로 대체됨 |
| 상호 엣지 | 1,598 (4.5%) | 신규 빌드 B-1 기준 2,106건 |

**신규 빌드 재측정 쿼리**:
```cypher
// 관계 총량
MATCH ()-[r]->() RETURN type(r) AS 관계, count(*) ORDER BY count(*) DESC;

// 역방향 대응률 · 같은방향 중복 (구 빌드 72.8% / 6.8% 대비)
MATCH (a:DocNumber)-[:EXAMINES_DECISION]->(b:DocNumber)
RETURN count(*) AS 전체,
       count(CASE WHEN (b)-[:ADOPTS_INTENT]->(a) THEN 1 END) AS 역방향대응,
       count(CASE WHEN (a)-[:ADOPTS_INTENT]->(b) THEN 1 END) AS 같은방향중복;

// ADOPTS 초과분의 정체 — EXAMINES 대응이 없는 ADOPTS 엣지 (실측: 10,728)
MATCH (a:DocNumber)-[:ADOPTS_INTENT]->(b:DocNumber)
WHERE NOT (b)-[:EXAMINES_DECISION]->(a)
RETURN count(*) AS 편측_참조결정;

// ★ §2-1 합집합이 실제로 추가할 엣지 수 — 실행 가치 판정의 핵심
MATCH (a:DocNumber)-[:EXAMINES_DECISION]->(b:DocNumber)
WHERE NOT (b)-[:ADOPTS_INTENT]->(a)
RETURN count(*) AS 편측_따른결정;
// 산술 추정: ADOPTS 5만 중 편측 10,728 → 역방향 대응 약 39,272쌍.
// EXAMINES 약 4만 이므로 편측_따른결정 ≈ 700~1,600 (3~4%) 예상.
// 구 빌드 27%와 대비 — 사실이면 §2-1의 실익은 미미(아래 판정표 참조).
```

**✅ §2-1 실행 판정 — 확정**: `편측_따른결정 = 1,545` (3.8%).

커버리지 보강 효과 자체는 미미(50,000 → 51,545, +3.1%)하나,
**§2-1은 EXAMINES 폐기(§1-5)의 전제 조건이므로 반드시 실행**한다 —
1,545건은 EXAMINES에만 존재하는 고유 정보이므로, §2-1 없이 폐기하면
그대로 소실된다.

| 관계 | 고유 정보 | 비율 |
|---|---|---|
| ADOPTS만 (편측 참조결정) | 10,728 | 21.5% |
| EXAMINES만 (편측 따른결정) | 1,545 | 3.8% |
| 양쪽 중복 | 약 39,272 | |

→ 확정 순서: **§2-1 실행(1,545 흡수) → 검증(편측_따른결정 = 0) →
EXAMINES 폐기 가능(정보 손실 0)**. 최종 상태는 ADOPTS_INTENT 단일
관계 약 51,545건.
| CaseName 노드 | 234,469 | 문서수보다 많음 = 문서당 복수 쟁점 |
| CaseName 단독/공유 | 207,076 (88.3%) / 27,393 (11.7%) | 최대공유 1,254 · 평균 1.238 |
| 심판↔판례 관계 | **0** | 공백 |
| 심판↔예규 관계 | **0** | 공백 |

> **✅ 신규 빌드 ETL 누락 해소**: 재구축 초기에 참조결정 67 / 따른결정 0
> 으로 로더가 누락됐던 문제는 **해결됨** — 현재 ADOPTS 약 5만 / EXAMINES
> 약 4만으로 구 빌드(29,420 / 35,315)보다 오히려 증가.
> 상시 대조: `MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC;`
>
> **⚠ 날짜 필드 — `reference_date`가 결정일자 정본**: 문서
> `[사건번호] 조심2020지1785 (2021.06.14)`의 괄호 날짜가 결정일이며
> 그래프 프로퍼티 `reference_date`가 이를 가리킨다(RESOLVES_TO의 시행본
> 시점 앵커로 이미 사용 중 = 신뢰 가능). **`reg_date`는 표본 전량 null**
> — 시간 조건은 전부 `reference_date` 기준으로 작성할 것.
>
> **사건번호의 연도 ≠ 결정연도**(접수연도임): 조심2020지1785는 2020년
> 접수·**2021.06.14 결정**. 사건번호 연도로 방향·병합을 판정하면
> 오분류가 발생한다(§1-2).
>
> 파생 연도 백필(집계·비교 편의) + 커버리지 확인:
> ```cypher
> CALL apoc.periodic.iterate(
>   "MATCH (d:DocNumber) WHERE d.reference_date IS NOT NULL RETURN d",
>   "SET d.ref_year = toInteger(left(toString(d.reference_date), 4))",
>   {batchSize: 10000});
>
> MATCH (d:DocNumber) RETURN count(*) AS 전체,
>        count(d.reference_date) AS 결정일보유, count(d.reg_date) AS reg_date보유;
> ```
>
> **⚠ 코드측 결함 (C12)**: `models/tools/gdb_cypher.py`가 두 필드를 혼용 —
> 법령 시점 앵커만 `d.reference_date`(351행), 나머지 12곳은 `d.reg_date`를
> "결정일"로 사용. reg_date가 비어 있으면 `count_cases`의 `date_from`
> 필터·정렬·선후 비교가 무력화된다. **12곳 전부 교체 필요.**

---

# §0. 선행 — DecisionType.canonical 백필

**웹 서비스 코드는 canonical을 만들지 않고 소비만 한다**
(`core/gdb/decision_canonical.py`가 단일 진실원). 소비 지점 3곳:
① GDB `dt.canonical IN $decision_canonicals` ② 질의 표면형 해석
③ VDB 필터 — Neo4j에서 canonical→names **런타임 로드**(없으면 결함 있는
구 수동사전으로 강등). **테스트 그래프에 미구축 시 결정필터·극성 관련
작업이 전부 침묵하므로 최우선.**

canonical 8종: `납세자_승 / 납세자_일부승 / 납세자_패 / 납세자_일부패 /
재조사 / 종결 / 기타 / 검색안됨` + 필터 제외 특수 `__CRIMINAL__ /
__PARTY_OTHER__ / __POLLUTED__`.

**관점**: canonical은 **납세자 축**이다. 원표기(name)는 심판원 절차 시점
(인용/기각), 법원 관용어(국승/국패), 처분청 시점(처분청 승소)이 혼재하며
canonical이 이를 하나의 축으로 번역한다. 국패=국가 패=**납세자_승**,
국승=**납세자_패** (직관과 반대 — 실수 잦은 지점).

```cypher
// (0) 현황
MATCH (dt:DecisionType) RETURN count(*) AS 전체, count(dt.canonical) AS 기구축;

// (1) 규칙 캐스케이드 — 코드 극성 규칙과 동일. 순서 중요, 멱등(NULL만 채움)
MATCH (dt:DecisionType) WHERE dt.canonical IS NULL
WITH dt, apoc.text.replace(dt.name, '[\\s\\u3000]+', '') AS n
SET dt.canonical = CASE
  WHEN n CONTAINS '재조사' THEN '재조사'
  WHEN n CONTAINS '취하' OR n CONTAINS '종결' THEN '종결'
  WHEN n CONTAINS '각하' THEN '납세자_패'
  WHEN n =~ '.*(처분청|과세관청|국가).*패.*'
    THEN CASE WHEN n CONTAINS '일부' THEN '납세자_일부승' ELSE '납세자_승' END
  WHEN n =~ '.*(처분청|과세관청|국가).*승.*'
    THEN CASE WHEN n CONTAINS '일부' THEN '납세자_일부패' ELSE '납세자_패' END
  WHEN n =~ '.*(납세자|원고|청구인).*승.*'
    THEN CASE WHEN n CONTAINS '일부' THEN '납세자_일부승' ELSE '납세자_승' END
  WHEN n =~ '.*(납세자|원고|청구인).*패.*'
    THEN CASE WHEN n CONTAINS '일부' THEN '납세자_일부패' ELSE '납세자_패' END
  WHEN n CONTAINS '국패' OR n CONTAINS '인용' OR n CONTAINS '취소'
    THEN CASE WHEN n CONTAINS '일부' THEN '납세자_일부승' ELSE '납세자_승' END
  WHEN n CONTAINS '국승' OR n CONTAINS '기각' THEN '납세자_패'
  WHEN n CONTAINS '경정' AND NOT n CONTAINS '경정청구' THEN '납세자_일부승'
  WHEN n = '검색안됨' THEN '검색안됨'
  ELSE NULL
END;

// (2) 잔여 분류 패치 — 실측 75종 기준
UNWIND ['유죄','일부유죄','무죄','집행유예','벌금형','실형','징역'] AS n
MATCH (dt:DecisionType {name: n}) SET dt.canonical = '__CRIMINAL__';

UNWIND ['저당권자패소','근저당권자패소','은행승소','은행패소','학교법인승소',
        '채무자승소','채무자승','채무자패소','채권자승소','채권자승',
        '기술신용승소','파산관제인패소','상대방패소','신청인승소','신청인패소',
        '재항고인패소'] AS n
MATCH (dt:DecisionType {name: n}) SET dt.canonical = '__PARTY_OTHER__';

UNWIND ['8년 자경','접대비 손금불산입','비업무용 부동산','부당행위','경정청구 거부처분',
        '실거래가액','배당','비지정기부금','사해행위','절차','양도','대여금','법인',
        '국징','O정'] AS n
MATCH (dt:DecisionType {name: n}) SET dt.canonical = '__POLLUTED__';

UNWIND ['화해','강제조정','조정성립','화해권고결정','소송종료선언'] AS n
MATCH (dt:DecisionType {name: n}) SET dt.canonical = '종결';

UNWIND ['기타','보류','보류 -23','판결','선고','무변론판결','1심','사건부참조',
        '상고불허가','재항고','일부','파기환송','파기환송(일부)','파기환송일부',
        '심리불속행'] AS n
MATCH (dt:DecisionType {name: n}) SET dt.canonical = '기타';

// (3) 원고 기준 승패 — ※ 주체 확인 후 실행 (아래 검증 참조)
UNWIND [{n:'일부패소',c:'납세자_일부패'},{n:'일부 패소',c:'납세자_일부패'},
        {n:'일부패',c:'납세자_일부패'},{n:'부분패소',c:'납세자_일부패'},
        {n:'일패',c:'납세자_일부패'},{n:'패소',c:'납세자_패'},
        {n:'전부패소',c:'납세자_패'},{n:'원소패소',c:'납세자_패'},
        {n:'일부승',c:'납세자_일부승'},{n:'승',c:'납세자_승'},
        {n:'국일부승',c:'납세자_일부패'}] AS m
MATCH (dt:DecisionType {name: m.n}) SET dt.canonical = m.c;

// (3-검증) '일부패소' 2,468건의 주체 확인 — plaintiff가 납세자인가
MATCH (d:DocNumber)-[:RESULTED_IN]->(:DecisionType {name:'일부패소'})
RETURN d.name, d.plaintiff, d.defendant LIMIT 10;

// (4) 피고 계열 38건 — defendant가 세무서장·지자체장이면 아래, 제3자면 __PARTY_OTHER__
UNWIND [{n:'피고승소',c:'납세자_패'},{n:'피고승',c:'납세자_패'},
        {n:'피고일부승소',c:'납세자_일부패'},{n:'피고패소',c:'납세자_승'}] AS m
MATCH (dt:DecisionType {name: m.n}) SET dt.canonical = m.c;

// (5) 검증
MATCH (dt:DecisionType) WHERE dt.canonical IS NULL RETURN count(*) AS 잔여NULL;
MATCH (dt:DecisionType)<-[:RESULTED_IN]-(d)
RETURN dt.canonical, count(d) AS 사건수 ORDER BY 사건수 DESC;
```

**지름길**: 운영 DB가 살아있다면 규칙 재현보다
`MATCH (dt:DecisionType) RETURN dt.name, dt.canonical` 덤프를 통째 이식하는
것이 가장 안전(불일치 리스크 0).

---

# §0-B. 데이터 품질 — 유령 노드·표기 이상

## 0-B-1. 연도 오타로 생성된 유령 노드

**실측 발견**: `국심1192서0547` — 1192년은 존재하지 않으며 `1992`의 오타.
인용 문서(국심1992서0829, 1992-07-02 결정)의 참조결정 기재 오타가 ETL을
통과해 **실체 없는 스텁 노드**를 생성했다. 이런 노드는 본문 없이 참조
기재로만 만들어지므로 `reference_date`·`summary`가 비어 있는 것이 특징.

> **⚠ 잘못된 접근 (기록용)**: `regexGroups(d.name,'\\d{4}')[0][0]` 로 첫
> 4자리를 연도로 읽는 방식은 **심판(조심/국심+4자리)에만 유효**하다.
> 판례는 `대전고등법원 93구1815`처럼 연도가 2자리이고 뒤 숫자가 일련번호,
> 예규는 `징세과-1709_2014.12.16.`처럼 문서번호라서 전부 오탐이 된다.
> 유형별로 분리해 검사할 것.

```cypher
// [심판] 접수연도 vs 결정연도 모순 — 오타 탐지의 정공법
//   접수보다 결정이 앞서거나(불가능), 5년 초과 격차(사실상 없음) = 오타
MATCH (d:DocNumber)
WHERE d.name =~ '^(조심|국심|감심)\\d{4}.*' AND d.reference_date IS NOT NULL
WITH d, toInteger(substring(d.name, 2, 4)) AS 접수연도,
     toInteger(left(toString(d.reference_date), 4)) AS 결정연도
WHERE 접수연도 > 결정연도 OR 결정연도 - 접수연도 > 5
RETURN d.name, 접수연도, 결정연도, 결정연도 - 접수연도 AS 차이,
       count{ (d)--() } AS 연결수
ORDER BY abs(결정연도 - 접수연도) DESC LIMIT 30;
// 실측 발견: 국심1955서3399 (결정일 1996-02-14, 41년 차 → 1995 오타)
//            국심1192서0547 (1192년 → 1992 오타)

// [심판] reference_date 조차 없는 경우 — 연도 범위만으로 판정
MATCH (d:DocNumber)
WHERE d.name =~ '^(조심|국심|감심)\\d{4}.*' AND d.reference_date IS NULL
WITH d, toInteger(substring(d.name, 2, 4)) AS y
WHERE y < 1970 OR y > 2026
RETURN d.name, y, count{ (d)--() } AS 연결수 LIMIT 20;

// [판례] 사건구분 한글 앞의 연도 토큰 (2자리는 19xx/20xx 환산)
MATCH (d:DocNumber) WHERE d.name CONTAINS '법원'
WITH d, apoc.text.regexGroups(d.name, '(\\d{2,4})[가-힣]{1,2}\\d+$') AS g
WHERE size(g) > 0
WITH d, toInteger(g[0][1]) AS raw
WITH d, CASE WHEN raw < 100
             THEN CASE WHEN raw >= 50 THEN 1900 + raw ELSE 2000 + raw END
             ELSE raw END AS 연도
WHERE 연도 < 1960 OR 연도 > 2026
RETURN d.name, 연도, count{ (d)--() } AS 연결수 LIMIT 20;

// [전 유형] 스텁 판별 — 자체 내용 없이 참조로만 존재하는 노드
//   ※ 판례·예규는 정상이어도 reference_date 가 null 인 경우가 있으므로
//     summary 부재와 단일 연결을 함께 볼 것
MATCH (d:DocNumber)
WHERE d.summary IS NULL
RETURN count(*) AS 본문없는노드,
       count(CASE WHEN count{ (d)--() } <= 1 THEN 1 END) AS 고립또는단일연결;
```

**처리 방침**: ①오타가 명백하고 정타 노드가 존재하면 관계를 정타로
재연결 후 스텁 삭제 ②판단 불가면 `d.stub = true` 표시만 하고 유지
(관계 정보 자체는 보존). 군집화·PageRank 전에 처리할 것 — 스텁이
차수 1짜리 노이즈 노드로 들어간다.

## 0-B-2. 사건번호 자릿수 변이

`국심1991서00333`(5자리) 처럼 일련번호 자릿수가 불규칙한 표기 존재.
동일 사건이 4자리/5자리 두 노드로 분리 적재됐을 수 있으므로 확인:

```cypher
MATCH (d:DocNumber) WHERE d.name =~ '.*[가-힣]\\d{5,}$'
RETURN d.name, count{ (d)--() } AS 연결수 ORDER BY 연결수 DESC LIMIT 20;

// 앞자리 0 제거 시 중복되는 쌍이 있는지 (동일 사건 분리 적재 탐지)
MATCH (d:DocNumber)
WITH d, apoc.text.regexGroups(d.name, '^(.*[가-힣])0*(\\d+)$') AS g
WHERE size(g) > 0
WITH g[0][1] + g[0][2] AS 정규화키, collect(d.name) AS 표기들
WHERE size(표기들) > 1
RETURN 정규화키, 표기들 LIMIT 20;
```

---

# §1. EXAMINES_DECISION / ADOPTS_INTENT 사용

## 1-1. 의미론 확정

도메인 정의 — **참조결정** = 본 사건 판단의 길잡이가 된 선례·기준 판례 /
**따른결정** = 그 법리·해석을 그대로 적용해 내려진 결정. 연도 대조 실측과
정확히 일치:

| 소스 필드 | 관계 | 방향 | 의미 | 명칭 |
|---|---|---|---|---|
| 참조결정 | `ADOPTS_INTENT` | 문서 → **과거** | 내가 인용·검토한 선례 | ⚠ 정확하나 "채택"은 과장 |
| 따른결정 | `EXAMINES_DECISION` | 문서 → **미래** | 나를 인용한 후속 결정 | ❌ 오류 → `FOLLOWED_BY` |

**ETL 매핑은 정상, 관계 이름 하나가 잘못 붙은 것.**
`EXAMINES_DECISION`("상급심이 하급심을 심리")은 **단심 기관인 조세심판원에
성립하지 않는 의미** — 심판↔심판에 심급 관계는 없고, 이건 **선례 인용망**.

**⚠ "채택/적용"이라는 관계명(ADOPTS_INTENT)은 실제보다 강한 의미를
함의한다.** 정의상 따른결정은 "참조 결정의 법리를 **그대로 적용**하여
내린 결론"이지만, 실물 반례가 존재(§2-5) — 조심2020지1785(2021, 취소)와
조심2023지0457(2023, 기각)은 동일 쟁점에서 **정반대 결론**인데도 1785의
따른결정에 0457이 등재됨. 즉 이 필드는 "동의·적용"이 아니라 **"후속
인용 발생"만을 기록**하는 것으로 보인다 — 인용했다고 결론까지 같이
채택했다는 보장은 없음(diverges 섹션이 이를 정면으로 다룸).

두 관계는 같은 **인용** 사건의 양면(X가 Y를 참조 = Y의 따른결정에 X)일
가능성이 높고 72.8%가 역방향 대응하나, **이는 실증적 상관관계이지
논리적 보장이 아니다** — "참조했다"가 반드시 상대의 "따른결정"에 반영
된다는 필연은 없고, 27.2%의 비대응이 전부 행정 누락인지 애초에 편측만
성립하는 관계인지 확인되지 않았다. 합집합(§2-1)은 이 가정을 전제로
한 **보강**이며, 신규 생성분은 `source` 태그로 원본과 구분해 신뢰도를
낮게 취급할 것(검증 전 확정 사실로 오인 금지).

## 1-2. 두 집단 혼재 — 인용 vs 동시결정군 (⚠ "병합" 판정 근거 미검증)

**⚠ 라벨 재검토 필요.** 앞서 "동년 18.5%(6,531)는 병합·관련 사건"이라고
단정했으나, 근거를 되짚으면 스크린샷 하나(조심2010부0240~0248의 조밀한
상호 인용 그림)에서 파생한 가설일 뿐 — 문서 내용으로 실제 공식 병합
여부를 확인한 적이 없다. 게다가 **병합**(하나의 절차로 묶인 청구,
통상 사건번호·결정문에 "(병합)" 명시)과 **관련**(별개 사건번호, 비슷한
시기 심리·상호 인용하나 공식 병합은 아님)은 다른 개념인데 혼용했다.
0240~0248은 **서로 다른 사건번호 9개**이므로 "병합"보다 "관련·동시결정군"
이 정확한 명칭일 가능성이 높다.

**"사건번호 연도" 판정은 오분류 확인됨**: 조심2020지1785(2021.06.14
결정)의 참조결정 조심2020지2188은 **접수연도는 같지만 명백한 선례
인용**(사건번호 연도는 접수연도). → 결정일(`reference_date`) 간격으로
전환.

**임계값은 실측으로 확정됨 — 7일** (당초 30일 추측은 오류, B-1 참조).
더 직접적인 신호는 **상호 엣지(1,598건, 4.5%)**: 정상적인 선례 인용은
옛것→새것 단방향인데(같은 쌍에서 양방향이면 "누가 누구의 선례"인지
순환이 생겨 이상함), 같이 심리된 관련 사건은 편집 단계에서 서로를
상호 참조로 링크했을 가능성이 높다 — day_gap보다 reciprocity가 더
직접적인 판별 신호. **두 신호를 교차 검증한 뒤 임계값을 확정한다.**

아래는 실행 순서대로: **(A) day_gap 계산(임계 없이 순수 간격만) →
(B) 교차검증·내용대조 → (C) 검증 결과로 rel_kind 확정.** (B)가 (A)의
결과값을 참조하므로 이 순서를 반드시 지킬 것 — 거꾸로 실행하면 day_gap
프로퍼티가 아직 없어 검증 쿼리가 전부 빈 결과를 반환한다.

```cypher
// (A-0) self-loop 제거 — 신규 빌드 ETL에 a→a 엣지 존재(자기참조 버그).
//       ✅ 실측 규모: EXAMINES_DECISION 기준 **38건**으로 경미.
//       ※ B-2 표본에서 30건 중 7건(23%)으로 보였던 것은 표본 편향 —
//         B-2 필터(상호 AND day_gap≤7)를 self-loop는 정의상 100% 만족하므로
//         과다 대표된 것. 모집단 비율이 아님(표본→모집단 투사 오류 주의).
//       영향은 미미하나 상호비율·PageRank·군집 가중을 왜곡하므로 제거.
MATCH (a:DocNumber)-[r]->(a) RETURN type(r) AS 관계, count(*) AS 건수;   // 규모 확인
MATCH (a:DocNumber)-[r:EXAMINES_DECISION|ADOPTS_INTENT]->(a) DELETE r;   // 제거
// ※ 신규 빌드 ETL에서 자기참조 필터를 넣어 재발 방지할 것

// (A) day_gap 계산만 — rel_kind 는 아직 정하지 않음(임계 미확정 상태).
//     EXAMINES_DECISION·ADOPTS_INTENT 둘 다 적용(병합 사건은 참조결정
//     필드에도 서로를 기재할 개연성 — 같은방향 중복 2,416건이 그 후보(B-3에서 검증),
//     미적용 시 diverges·법리뿌리에 오염됨)
CALL apoc.periodic.iterate(
  "MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
   WHERE a.reference_date IS NOT NULL AND b.reference_date IS NOT NULL
   RETURN a, r, b",
  "SET r.day_gap = duration.inDays(date(a.reference_date), date(b.reference_date)).days",
  {batchSize: 5000});

CALL apoc.periodic.iterate(
  "MATCH (a:DocNumber)-[r:ADOPTS_INTENT]->(b:DocNumber)
   WHERE a.reference_date IS NOT NULL AND b.reference_date IS NOT NULL
   RETURN a, r, b",
  "SET r.day_gap = duration.inDays(date(a.reference_date), date(b.reference_date)).days",
  {batchSize: 5000});

// day_gap 커버리지 확인
MATCH ()-[r:EXAMINES_DECISION|ADOPTS_INTENT]->()
RETURN type(r), count(r) AS 전체, count(r.day_gap) AS day_gap보유;
```

```cypher
// (B-1) day_gap 구간별 상호엣지 비율 — day_gap이 실제 판별자인지 검증
MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
WHERE r.day_gap IS NOT NULL
WITH r, abs(r.day_gap) AS gap, EXISTS { (b)-[:EXAMINES_DECISION]->(a) } AS 상호
WITH CASE WHEN gap = 0 THEN '동일자' WHEN gap <= 7 THEN '1주내'
          WHEN gap <= 30 THEN '1개월내' WHEN gap <= 365 THEN '1년내'
          ELSE '1년초과' END AS 구간, 상호, count(*) AS n
RETURN 구간, sum(CASE WHEN 상호 THEN n ELSE 0 END) AS 상호건, sum(n) AS 전체,
       toFloat(sum(CASE WHEN 상호 THEN n ELSE 0 END))/sum(n) AS 상호비율
ORDER BY 구간;
// ✅ 확정 (**신규 빌드, self-loop 제거 후 재측정**) — 임계 = **7일**:
//   동일자 49.7%(1466/2947) · 1주내 47.2%(118/250) · 1개월내 1.7%(10/588)
//   · 1년내 1.1%(104/9466) · 1년초과 1.3%(370/27580)
//   → 1주내→1개월내에서 29배 급락, 1개월 이후는 1.1~1.3% 배경 노이즈.
//   → 동일자·1주내는 배경 대비 약 30~45배 = 명백히 다른 집단.
//   → **병합 후보 = 동일자 2,947 + 1주내 250 = 3,197건**
//   ※ self-loop 제거 영향은 38건뿐(동일자 2,985→2,947)으로 미미 — 판정 불변.
//   ※ 당초 임계 30일은 오류였음 — 1개월내 588건(상호 1.7% = 배경 수준,
//     즉 일반 인용)을 병합으로 오분류했을 것.

// (B-2) 병합 vs 관련 최종 판별 — day_gap≤후보임계 + 상호 표본 20~30건을
// 뽑아 청구인명 동일 여부·사건번호의 "(병합)" 표기 유무·동일 처분
// 기초사실 여부를 원문 직접 대조
MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
WHERE abs(r.day_gap) <= 7 AND EXISTS { (b)-[:EXAMINES_DECISION]->(a) }
RETURN a.name, b.name, r.day_gap LIMIT 30;
// → 원문 대조: 청구인 동일(병합 강한 신호) / 상이하나 같은 쟁점(관련) /
//   무관한 우연 근접(재분류 필요)
//
// ✅ B-2 실측 결과 (표본 30건) — 병합 가설 강하게 지지:
//   ① self-loop 7건 (a=b) → ETL 버그, (A-0)에서 제거
//   ② 완전 클리크: 국심1991서2678·2679·2680 이 서로를 전부 가리킴,
//      전건 day_gap=0 → 동시결정 형제의 전형
//   ③ 연번 다발: 국심1992서0179 → 0276·0277·0278…0285 (사건번호 연속),
//      같은 날 결정 → 동일 처분에 대한 다수 청구인의 병합 청구 형태
//   ④ 교차연도 근접: 국심1991서2680 → 국심1992서0179 (day_gap=1)
//      접수연도는 다르나 결정은 하루 차 → 접수연도 기준의 한계 재확인
//   → 원문 확정은 2678~2680 중 1건만 열어 청구인·"(병합)" 표기 확인하면 충분

// (B-3) 같은방향 중복 2,416건(6.8%)의 정체 — 병합 가설의 직접 증거
//
// ※ 역방향 대응(정상 72.8%)과 혼동 금지:
//    역방향 = (a)-[EXAMINES]->(b) + (b)-[ADOPTS]->(a)  ← 출발점이 서로 다름.
//             같은 인용을 두 문서가 각자 기록. 조심2020지1785↔2023지0457이 이 경우.
//    같은방향 = (a)-[EXAMINES]->(b) + (a)-[ADOPTS]->(b) ← 출발점이 둘 다 a.
//             a 문서 한 장의 참조결정·따른결정 칸에 b가 동시 등재됨 =
//             "내가 b를 참조" + "b가 나를 따름"이 동시 성립 → 시간상 모순.
//
// 가설: 같이 결정된 형제 사건은 선후가 없어 편집자가 양쪽 칸에 다 기재.
// day_gap이 0 근처에 몰리면 병합 가설 강화, 흩어지면 단순 기재 오류.
MATCH (a:DocNumber)-[e:EXAMINES_DECISION]->(b:DocNumber)
WHERE (a)-[:ADOPTS_INTENT]->(b)
WITH CASE WHEN e.day_gap IS NULL THEN '날짜없음'
          WHEN abs(e.day_gap) = 0 THEN '동일자'
          WHEN abs(e.day_gap) <= 7 THEN '1주내'
          WHEN abs(e.day_gap) <= 30 THEN '1개월내'
          ELSE '1개월초과' END AS 구간, count(*) AS 건수
RETURN 구간, 건수 ORDER BY 건수 DESC;

// 표본 — 동일자 건이 실제 병합인지 원문 대조 (청구인·"(병합)" 표기)
MATCH (a:DocNumber)-[e:EXAMINES_DECISION]->(b:DocNumber)
WHERE (a)-[:ADOPTS_INTENT]->(b) AND coalesce(abs(e.day_gap), 999) = 0
  AND a <> b                                    // self-loop 배제
RETURN a.name, b.name, e.day_gap LIMIT 10;

// ✅ B-3 실측 (신규 빌드, self-loop 미제거): 총 2,984건
//   동일자 2,328(78.0%) · 1주내 114(3.8%) · 1개월내 12(0.4%) · 1개월초과 530(17.8%)
//   → 78%가 동일자 = 병합 가설 지지. 단 self-loop 포함 수치이므로 (A-0) 후 재측정.
//   → 1개월초과 530건은 별개 문제: 시간이 벌어졌는데 참조·따른결정 양쪽
//     등재 = 진짜 기재 오류 후보 → (C)의 `mutual_far` 플래그 대상.
```

```cypher
// (C) rel_kind 확정 — 임계 **7일** (B-1 실측으로 확정, 당초 30일은 오류)
CALL apoc.periodic.iterate(
  "MATCH ()-[r:EXAMINES_DECISION|ADOPTS_INTENT]->() WHERE r.day_gap IS NOT NULL RETURN r",
  "SET r.rel_kind = CASE WHEN abs(r.day_gap) <= 7 THEN 'merged' ELSE 'citation' END",
  {batchSize: 5000});

// 이상치 플래그 — 결정일이 1개월 넘게 벌어졌는데 상호 인용(474건 추정:
// 1년내 104 + 1년초과 370). A가 B의 선례이면서 동시에 B가 A의 선례일 수
// 없으므로 데이터 품질 이슈 후보. 병합으로 넣지 말고 별도 표시.
MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
WHERE abs(coalesce(r.day_gap, 0)) > 30 AND EXISTS { (b)-[:EXAMINES_DECISION]->(a) }
SET r.anomaly = 'mutual_far';

// 평행(중복) 엣지 점검 — 상시 위생 점검용.
// ※ 당초 "B-1 합계 40,869 > 총량 35,315" 를 중복 엣지로 의심했으나,
//   40,869는 신규 빌드 · 35,315는 구 빌드 값으로 **빌드가 달랐던 것**.
//   중복 엣지 문제 아님(오탐). 다만 GDS 가중 왜곡 방지를 위해 확인은 유효:
MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
WITH a, b, count(r) AS n WHERE n > 1
RETURN count(*) AS 중복쌍수, sum(n - 1) AS 초과엣지, max(n) AS 최대중복;

// 중복 발견 시에만 정리 (한 쌍당 1개 유지)
// MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
// WITH a, b, collect(r) AS rs WHERE size(rs) > 1
// FOREACH (x IN rs[1..] | DELETE x);

// reference_date 결측 엣지는 day_gap 자체가 없어 미분류 — citation 폴백(안전)
MATCH ()-[r:EXAMINES_DECISION|ADOPTS_INTENT]->() WHERE r.rel_kind IS NULL
RETURN type(r), count(*) AS 미분류;
```

**실물 검증 표본** (업로드 2건으로 상호 대응 확인 완료):

| 문서 | 결정일 | 참조결정 | 따른결정 |
|---|---|---|---|
| 조심2020지1785 | 2021.06.14 | 조심2020지2188 | 조심2022지0273 / **조심2023지0457** / 조심2024지3603 |
| 조심2023지0457 | 2023.12.22 | **조심2020지1785** / 조심2021지1914 / 조심2021지0787 / 조심2022지0385 | (없음) |

→ A의 따른결정에 B, B의 참조결정에 A — 같은 인용을 양쪽에서 기록하는
구조가 실물로 확인. 그래프 기대 형태:
`(1785)-[:EXAMINES_DECISION]->(0457)` + `(0457)-[:ADOPTS_INTENT]->(1785)`

```cypher
// 이 쌍으로 적재 정합성 직접 확인
MATCH (a:DocNumber {name:'조심2020지1785'})-[r]-(b:DocNumber {name:'조심2023지0457'})
RETURN type(r), startNode(r).name AS 시작, endNode(r).name AS 끝;
```

## 1-3. 활용 쿼리

```cypher
// (a) 이 사건이 참조한 선례 — 정방향 ADOPTS_INTENT (병합 형제 제외)
MATCH (d:DocNumber {name:$doc_number})-[r:ADOPTS_INTENT]->(p:DocNumber)
WHERE coalesce(r.rel_kind, 'citation') = 'citation'
OPTIONAL MATCH (p)-[:RESULTED_IN]->(dt:DecisionType)
RETURN p.name AS 선례, dt.canonical AS 결과;

// (b) 이 결정을 따른 후속 결정 — 정방향 EXAMINES_DECISION (citation만)
MATCH (d:DocNumber {name:$doc_number})-[r:EXAMINES_DECISION]->(f:DocNumber)
WHERE coalesce(r.rel_kind,'citation') = 'citation'
RETURN f.name AS 후속결정 ORDER BY f.name;

// (c) 법리 뿌리 체인 — 선례의 선례까지 (병합 형제가 체인에 끼지 않게 rel_kind 필터)
MATCH path = (d:DocNumber {name:$doc_number})-[:ADOPTS_INTENT*1..3]->(root:DocNumber)
WHERE NOT (root)-[:ADOPTS_INTENT]->()
  AND all(r IN relationships(path) WHERE coalesce(r.rel_kind,'citation') = 'citation')
RETURN [n IN nodes(path) | n.name] AS 법리전파체인, length(path) AS 깊이
ORDER BY 깊이 DESC LIMIT 5;

// (d) 병합·관련 사건 (rel_kind=merged — 결정일 간격 기준)
MATCH (d:DocNumber {name:$doc_number})-[r:EXAMINES_DECISION]-(x:DocNumber)
WHERE r.rel_kind = 'merged'
RETURN x.name AS 관련사건;

// (e) 영향력 순위 — 피인용 기반 (검증 완료, 아래 참조)
MATCH (l:DocNumber)-[r:EXAMINES_DECISION]->(f:DocNumber)
WHERE coalesce(r.rel_kind,'citation') = 'citation'
WITH l, count(f) AS 후년인용수
ORDER BY 후년인용수 DESC LIMIT 20
OPTIONAL MATCH (l)-[:NAMED_AS]->(cn:CaseName)
RETURN l.name AS 선례, cn.name AS 제목, 후년인용수;
```

**(e) 영향력 지표 검증 완료** — 인위적 상한·병합 클리크 아님이 확인됨:
① 상위 20건 중 13건이 동년 0건, 전량 후년 인용
② 분포가 매끄러운 heavy-tail(36:6 → 34:8 → 29:12 → 18:20 → 14:40 →
   10:73 → 8:128 → 6:250) — 상한이 있으면 36에 무더기가 쌓여야 하나 없음
→ 편집자가 직접 기록한 명시적 영향력 신호이므로 ArticleRank 추정치보다
신뢰도 높음. `authority_ranking` op의 1차 근거로 승격 가능.

## 1-4. 코드 영향 (2026-07 패치 반영 현황)

| 항목 | 구 현상 | 상태 |
|---|---|---|
| `examines_lineage` | 후속 인용을 "전심 계보/상급심"으로 제시(오답) | ✅ **반영 완료** — 선례/후속/병합 구분 컬럼의 인용·관련 흐름 종합으로 재작성 |
| `citing_cases` | 선례·후속 혼재를 전부 "후속"으로 오라벨 | ✅ **반영 완료** — 후속 전용 합집합(EXAMINES 정방향 ∪ ADOPTS 역방향) |
| `adopts_intent_chain` | 정상 작동 | 유지 |
| 렌더·프롬프트 | "상급심/전심" 표현 + 국패/canonical 노출 | ✅ **반영 완료** — Cypher CASE 번역 6곳 + 프롬프트 번역표·표현 금지 |
| `authority_ranking` | ArticleRank 단독 | ✅ **반영 완료** — 후속적용수(rel_kind=citation) 1차 정렬 + pagerank 폴백 |
| `reg_date` 혼용 (C12) | 12곳이 빈 필드로 날짜 조건 무력화 | ⏳ **미반영** — reference_date로 교체 예정 |
| op 개명 (`FOLLOWED_BY` 등) | — | 🔒 **보류 확정**(§1-5) — 원천에 없는 개념 명명 회피 |
| `adopts_intent_chain` 시간 단조성 (C14) | 시간 역행 인용 체인이 답변에 노출 가능 | ⏳ **미반영** — §7-1 제약 추가 필요 |

상세: 변경사항_관계의미론_교정.md / 전체_변경사항_원본대비.md 참조.

## 1-5. 신규 빌드 권장 구조

```
(새것)-[:ADOPTS_INTENT]->(옛것)     // 인용 — 참조결정 + 따른결정 역삽입 합집합
(a)-[:RELATED_CASE]-(b)             // 병합·관련 — 결정일 간격(day_gap) 기준
```
`EXAMINES_DECISION`은 역방향 중복이므로 폐기, `<-[:ADOPTS_INTENT]-` 역순회로
대체. 합집합으로 커버리지 +27%.

**🔒 결정 (2026-07): 폐기·개명 모두 보류, 현행 2관계 유지.**
사유 — `FOLLOWED_BY` 같은 명칭은 **원천 데이터에 정의가 없는 개념**이다.
원천 필드는 참조결정·따른결정 둘뿐이며, 이를 "follow"라는 관계로 명명하는
것은 해석이지 사실이 아니다. 정의되지 않은 개념을 관계명으로 심으면
그래프가 원천에 없는 주장을 하게 되므로, **관계명은 원천 필드에 대응하는
현행 유지**(EXAMINES_DECISION = 따른결정, ADOPTS_INTENT = 참조결정).
의미 왜곡은 **관계명이 아니라 렌더·프롬프트 층에서 교정**한다(§1-4 완료분).

아래 대응표는 **장래 폐기를 재검토할 경우에만** 적용. 현재는 비활성.

**⚠ 폐기 채택 시 필수 대응표** — 본 문서와 현행 코드 패치의 다수 쿼리가
EXAMINES_DECISION을 순회하므로, 신규 빌드에서 폐기를 채택하면 아래를
함께 교체해야 함(안 하면 해당 기능 전부 빈 결과):

| 현행 (EXAMINES 기반) | 폐기 시 대체 |
|---|---|
| §1-3(b)(e)·§5-4 후속/피인용: `(d)-[:EXAMINES_DECISION{citation}]->()` | `(d)<-[:ADOPTS_INTENT]-()` (merged 제외) |
| §1-3(d)·§2-2 병합: `-[:EXAMINES_DECISION{merged}]-` | `-[:RELATED_CASE]-` |
| 코드 `examines_lineage`·`citing_cases`·`authority_ranking` 후속적용수 | 동일 원리로 재수정 (op 개명과 함께 일괄) |

**결정 전까지는 구/신 어느 빌드든 동작하도록 현행 코드는 EXAMINES 기준 +
coalesce 폴백을 유지 중** — 두 관계를 병존시키는 선택(폐기 안 함, 이름만
`FOLLOWED_BY`로 교체)도 유효하며, 이 경우 대응표 불요.

---

# §2. 심판 ↔ 심판 관계 생성

**현황: 이미 풍부**(37.5%). 신규 구축이 아니라 **보강·분리**가 과제.

## 2-1. 합집합 보강 — **✅ 실행 완료 (2026-07)**

**✅ 실측 확정**: 편측 참조결정 10,728(21.5%) / 편측 따른결정 1,545(3.8%).
신규 빌드는 **참조결정이 주 기록, 따른결정이 그 부분 역인덱스**인 비대칭
구조이며 **EXAMINES는 ADOPTS 역방향에 96.2% 포함**된다.

**목적 변천**: ①"커버리지 +27% 보강"(구 빌드) → 신규에선 +3.0%
②"EXAMINES 폐기 전제" → 폐기 보류(§1-5)로 소멸 ③"코드가 질의 시점에
합집합하므로 불요" — 다만 **실제로 실행됨**. 데이터 층에서도 합쳐두면
GDS 프로젝션(§5-2)에서 편측 누락이 사라지는 이점이 있어 무해하다.

### ✅ 실행 결과 (실측)

| 항목 | 값 | 해석 |
|---|---|---|
| 신규 생성(`source='followed_inverse'`) | **1,134** | 전체 citation 37,634 대비 3.01% |
| 기존(`source=null`) | 50,005 | 원본 참조결정 |
| ADOPTS 합계 | **51,139** | |
| 잔여 편측 따른결정 | **411** | **버그 아님** — `rel_kind='merged'` 이라 필터에서 정상 제외 |

**1,134 + 411 = 1,545** — 사전 측정한 편측 따른결정과 정확히 일치.
411건은 병합 관계이므로 인용(ADOPTS)으로 합치면 안 되며, **§2-2
RELATED_CASE의 대상**이다.

**🎯 교차검증 성공**: 전체 citation 37,634 → merged = 40,831 − 37,634 =
**3,197건**으로, B-1에서 예측한 병합 후보(동일자 2,947 + 1주내 250 =
3,197)와 **정확히 일치**. rel_kind 분류와 임계 7일이 의도대로 적용됨을
독립적으로 확인.

현행 `examines_lineage`·`citing_cases`는 다음과 같이 두 관계를 합집합
순회한다 — 편측 1,545건이 **데이터 병합 없이도 이미 결과에 포함**된다:
```
선례 = (d)-[:ADOPTS_INTENT]->(x)  ∪  (x)-[:EXAMINES_DECISION]->(d)
후속 = (d)-[:EXAMINES_DECISION]->(x)  ∪  (x)-[:ADOPTS_INTENT]->(d)
```

**단 하나의 예외 — GDS 프로젝션**(§5-2): 이중 카운트 회피를 위해 ADOPTS만
투영하면 편측 1,545건(약 3%)이 군집화에서 누락된다. 세 선택지 중 택일:

| 방법 | 결과 |
|---|---|
| ADOPTS만 투영 | 1,545건(3%) 누락 — 실용상 허용 가능 |
| 둘 다 투영 | 겹치는 39,272쌍 가중 2배 — 왜곡, 비권장 |
| **Cypher 프로젝션 합집합+중복제거** | 정확 (§5-2 참조) |

아래 병합 쿼리는 **참고용으로만 보존**(실행 불요):

**전제 재확인**: "a의 따른결정에 b가 있으면 b의 참조결정에도 a가
있어야 한다"는 **실증 상관관계(구 빌드 72.8%)이지 논리적 보장이 아니다**
(§1-1). 남은 비대응이 행정 누락인지, 애초에 편측(예: 스치듯 언급 vs 정식
참조)만 성립하는 관계인지 미확인. 따라서 본 보강은 **가정에 기반한
추론이며, 표본 검증 없이 원본과 동일한 신뢰도로 쓰지 말 것.**

한쪽 문서에만 기재된 인용을 상대 필드에서 건짐. `WHERE NOT (b)-[:ADOPTS_
INTENT]->(a)`가 이미 채워진 72.8%를 걸러내려는 의도이나, MERGE 자체가
멱등이라 **이 WHERE 없이 돌려도 데이터는 틀리지 않는다** — 다만 25,716건
(72.8%)을 매번 재확인하는 낭비 순회가 생기므로 조건절을 유지. 신규 생성
여부는 **소스 태그(`followed_inverse`)로 사후 검증**할 것 — 전건이 이미
있었는지 실제로 27%가 새로 붙었는지는 실행 전 조건절만으론 보장 안 됨.

```cypher
// 따른결정을 역방향으로 ADOPTS_INTENT에 합류 (누락분만 채워짐 — MERGE 멱등)
CALL apoc.periodic.iterate(
  "MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
   WHERE coalesce(r.rel_kind,'citation') = 'citation'
     AND NOT (b)-[:ADOPTS_INTENT]->(a)
   RETURN a, b",
  "MERGE (b)-[n:ADOPTS_INTENT]->(a) SET n.source = 'followed_inverse'",
  {batchSize: 5000});

// 검증 — 신규 생성분이 실제로 ~27%대인지 확인 (기대와 다르면 rel_kind 분류나
// 역방향 대응 전제 자체를 재점검)
MATCH ()-[r:ADOPTS_INTENT]->() RETURN r.source, count(*) ORDER BY count(*) DESC;
MATCH ()-[r:ADOPTS_INTENT {source:'followed_inverse'}]->()
WITH count(r) AS 신규
MATCH ()-[r2:EXAMINES_DECISION]->() WHERE coalesce(r2.rel_kind,'citation')='citation'
WITH 신규, count(r2) AS 전체citation
RETURN 신규, 전체citation, toFloat(신규) / 전체citation AS 신규비율;
// 기대: 신규 ≈ 1,545 (실측 편측_따른결정 값)

// ★ 합집합 완결 판정 — **citation 한정**으로 볼 것.
// (merged 엣지는 인용이 아니므로 ADOPTS로 합치지 않는 게 정상 — 411건이
//  여기 해당. 전체 기준으로 0을 기대하면 영원히 충족되지 않음)
MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
WHERE coalesce(r.rel_kind,'citation') = 'citation'
  AND NOT (b)-[:ADOPTS_INTENT]->(a)
RETURN count(*) AS 잔여_citation_편측;   // 0 기대 (실행 후 확인)

// 참고: 전체 기준 잔여 = 411 (전량 rel_kind='merged') — §2-2에서 처리
```

**표본 검증 (필수, 규모 확정 전)** — followed_inverse 20~30건을 뽑아
원문(HWP/PDF)에서 실제로 참조결정 필드에 해당 사건이 있었는지(누락
확인) 또는 없었는지(편측 관계 확인) 직접 대조:

```cypher
// ⚠ ORDER BY 시차 DESC 로 뽑으면 극단값 30건만 보게 됨 — 대표성 없음.
//   원문 대조 표본은 반드시 무작위 또는 주요 구간(1~3년)에서 추출할 것.
MATCH (c:DocNumber)-[r:ADOPTS_INTENT {source:'followed_inverse'}]->(p:DocNumber)
WITH c, p, rand() AS rnd ORDER BY rnd LIMIT 20
RETURN c.name AS 인용문서, p.name AS 선례, c.reference_date AS 인용일;
// → '인용문서'의 원문 [참조결정] 칸에 '선례'가 있는지 확인:
//    있음 → ETL 누락이었음(보강이 옳음, 통상 신뢰도로 승격)
//    없음 → 편측 관계(confidence 0.6 등으로 낮추고 정밀 분석에서 제외)
```

### ✅ 보강분 1,134건 성격 규명 (실측)

**시차 분포** — 80%가 3년 이내로, 통상적 인용 지연 범위:

| 구간 | 건수 | 비율 |
|---|---|---|
| 1년내 | 457 | 40.3% |
| 1~3년 | 453 | 39.9% |
| 3~5년 | 155 | 13.7% |
| 5~10년 | 57 | 5.0% |
| 10년+ | 12 | 1.1% |

**선례 집중도** — 상위 35개 선례가 250건(22%)을 차지하나 나머지 884건은
1~3건씩 넓게 분산. 소수 리딩케이스 편중 아님.

**판정: 통상적 기재 누락**. 인용 문서 작성자가 [참조결정] 칸에 적지
않았을 뿐 실제 참조 관계로 보이며, **보강 1,134건은 통상 신뢰도로
사용 가능**(원문 표본 대조로 최종 확인 권장 — 표본은 1년내·1~3년
구간에서 추출).

> **⚠ 분석 교훈 (2회 반복된 오류)**: 최장 시차 30건(7,181~2,598일)만
> 보고 "편집부가 리딩케이스를 수십 년간 사후 갱신한다"는 가설을 세웠으나,
> 전체 분포 확인 결과 10년+ 는 **1.1%(12건)** 에 불과해 기각됨.
> 앞서 self-loop 규모를 표본 23%로 추정했다가 실제 38건이었던 것과 동일한
> **필터링·정렬된 표본을 모집단에 투사하는 오류**. 표본 추출 시
> `ORDER BY` 극단 정렬 대신 `rand()` 또는 구간 층화를 기본으로 할 것.

누락으로 확인되면 통상 신뢰도로 승격, 편측 관계로 확인되면(원문에 정말
없음) **`confidence: 0.6` 등으로 낮춰 두고 diverges·법리뿌리(§1-3c)
같은 정밀 분석에서는 제외 옵션 유지.**

## 2-2. 관련사건 분리 — ✅ 완료 (RELATED_CASE 2,405 무방향 엣지)

**처리 대상 확정**: rel_kind='merged' 3,197건(EXAMINES 기준). 이 중
**411건은 ADOPTS 역방향 대응이 없는 편측**으로, §2-1에서 정상 제외된
것들 — 인용이 아니라 병합 관계이므로 여기서 RELATED_CASE로 흡수한다.

**§1-2 교차검증(day_gap×상호성) 및 내용 표본 대조 전에는 실행 보류
권장.** `rel_kind='merged'`는 현재 "동시결정·상호인용 후보"일 뿐, 공식
병합인지 단순 관련 사건인지 미확정. 검증 후 실행:

```cypher
// RELATED_CASE 로 승격 (rel_kind='merged' — §1-2 검증 통과분만)
CALL apoc.periodic.iterate(
  "MATCH (a:DocNumber)-[r:EXAMINES_DECISION]->(b:DocNumber)
   WHERE r.rel_kind = 'merged' RETURN a, b, r",
  "MERGE (a)-[n:RELATED_CASE]-(b) SET n.source = 'day_gap_merged', n.day_gap = r.day_gap",
  {batchSize: 5000});
```

### ✅ 실행·검증 완료 (2026-07)

| 항목 | 결과 | 판정 |
|---|---|---|
| 무방향 엣지 | **2,405** | 방향 엣지 3,197 → 상호쌍 792 접힘 + 단방향 1,613 = 2,405 ✓ **B-1 상호엣지(1,584)와 정확히 일치** |
| 최대 차수 | 26 | 허브 오염 없음(수백 규모 무관 사건 묶임 아님) |
| 세목 순도 | **100%** | 차수 20+ 전 문서가 단일 세목(법인 / 부가) |

**병합군의 실제 성격 규명** — 사건번호 패턴이 결정적:

```
[2011년 법인세군, 차수 20]
조심2011광1872·1873 / 구1856·1857·1868 / 부1854·1855·1867·1869
      / 전1863·1865 / 중1858·1861·1862
→ 일련번호 1854~1873 연속 구간, 지역코드(광·구·부·전·중)는 전부 상이
```

즉 **지역 단위 병합이 아니라 본부 차원의 동일쟁점 일괄처리** — 전국
각 지방국세청에 흩어져 접수된 같은 쟁점 청구를 한꺼번에 심리·결정한
형태(2012년 부가세군 0595~0601도 동일 패턴). day_gap ≤ 7 기준이 이
구조를 정확히 포착하고 있으며, **세목·처분청 추가 조건은 불필요**.

※ 차수 20인데 목록에 14문서만 보이는 것은 조회 필터(`deg >= 20`) 때문 —
실제 군집은 21개 이상. 정확한 크기 분포는 WCC로:

```cypher
CALL gds.graph.project('rel_check', 'DocNumber',
  {RELATED_CASE: {orientation:'UNDIRECTED'}});
CALL gds.wcc.stream('rel_check') YIELD nodeId, componentId
WITH componentId, count(*) AS 크기
RETURN 크기, count(*) AS 군집수, sum(크기) AS 문서합 ORDER BY 크기 DESC LIMIT 20;
CALL gds.graph.drop('rel_check');
```

**LLM 직조 시 표현**: "병합 사건"보다 **"같은 쟁점으로 함께 심리·결정된
관련 사건"**이 실태에 부합(전국 각지 접수분의 일괄처리이므로 좁은 의미의
소송법상 병합과는 다름).

**유지보수 원칙**: `rel_kind`가 단일 진실원, `RELATED_CASE`는 파생물.
day_gap 임계를 변경해 재분류할 경우 RELATED_CASE를 **삭제 후 재생성**할 것
(낡은 엣지 잔존 방지).
```cypher
MATCH ()-[r:RELATED_CASE]-() DELETE r;   // 재분류 시 선행
```

## 2-3. 쟁점(Issue) 매개 — ✅ 축 A 확정 (하이브리드 2층)

### 실측 결론 (2026-07)

**문자열 클러스터링 단독으로는 축 A가 성립하지 않는다.** 정규화 후에도
실체 제목 262,805개 / 문서 313,463건으로 **제목당 1.19문서** — 87%가
고유 제목. support≥3 클러스터는 8,533개(문서 50,261, 16%)에 그치고,
그중 **유형교차는 741개(8.7%)뿐**:

| 유형수 | 쟁점수 | 문서합 |
|---|---|---|
| 1 (동일 유형만) | 7,792 | 42,242 |
| 2 | 606 | 5,123 |
| **3 (전 유형 관통)** | **135** | 2,896 |

**반면 키워드 층은 20개 전량이 3유형을 관통** — 축 A의 실질 매개는
문장이 아니라 **쟁점 어휘**임이 확정:

| 쟁점 | 문서수 | 유형 |
|---|---|---|
| 세금계산서 | 14,396 | 판례·심판·예규 |
| 비과세 | 13,386 | 판례·심판·예규 |
| 매입세액 | 8,262 | 판례·심판·예규 |
| 가산세 | 7,291 | 판례·심판·예규 |
| 취득가액 | 6,885 | 판례·심판·예규 |
| … (20개 전량 3유형) | … | … |
| 부속토지 | 1,464 | 판례·심판·예규 |
| 고급주택 | 513 | 판례·심판·예규 |

### 설계 — 2층 하이브리드

| 층 | 방식 | 규모 | 성격 |
|---|---|---|---|
| 1층 | 문자열 클러스터(support≥3) | 8,533 쟁점 / 50,261 링크 | 정밀·협소 |
| 2층 | **키워드 사전 매칭** | 20~80 쟁점 / 대부분 커버 | 재현·광범 |

**⚠ 허브 문제와 대응**: 상위 쟁점은 1만 건대라 **단독 검색키로 부적합**.
또한 쟁점어 간 변별력 차이가 큼(범용: 과세표준·가산세 / 특정: 자경농지·
부속토지·고급주택). → **IDF 특이도 부여 + 조합 서명**으로 정밀도 확보.

```cypher
// (1) 사전 매칭 — 키워드는 토큰 빈도 분석에서 자동 추출(수기 아님)
:param kw => ['취득가액','필요경비','양도차익','양도가액','기준시가','실지거래가액',
              '취득시기','양도시기','1세대1주택','비과세','세금계산서','매입세액',
              '영세율','공급시기','명의신탁','상속재산','피상속인','자경농지',
              '가산세','과세표준','납세의무자','경정청구','부당행위','고급주택',
              '부속토지','사업자등록','소득구분','과세대상'];

CALL apoc.periodic.iterate(
  "UNWIND $kw AS k
   MATCH (cn:CaseName) WHERE cn.issue_class='substantive' AND cn.name CONTAINS k
   MATCH (d:DocNumber)-[:NAMED_AS]->(cn)
   RETURN DISTINCT d, k",
  "MERGE (i:Issue {name: k}) ON CREATE SET i.source='keyword'
   MERGE (d)-[h:HAS_ISSUE]->(i) ON CREATE SET h.source='keyword', h.confidence=0.85",
  {batchSize: 2000, params: {kw: $kw}});

// (2) IDF 특이도 — 범용어 자동 감쇠
MATCH (i:Issue)<-[:HAS_ISSUE]-() WITH i, count(*) AS df
MATCH (d:DocNumber) WITH i, df, count(d) AS N
SET i.support = df, i.idf = log(1.0 * N / df);

// (3) 조합 서명 검색 — 실질 검색 단위(단일 키워드는 너무 넓음)
MATCH (d:DocNumber {name:$doc_number})-[:HAS_ISSUE]->(i:Issue)
WITH d, collect(i) AS issues
MATCH (x:DocNumber)-[:HAS_ISSUE]->(i2:Issue)
WHERE i2 IN issues AND x <> d
WITH d, x, count(DISTINCT i2) AS 공유쟁점수, sum(i2.idf) AS 가중합
WHERE 공유쟁점수 >= 2
OPTIONAL MATCH (x)-[:CONCERNS_TAX]->(t:TaxItem)<-[:CONCERNS_TAX]-(d)
RETURN x.name, 공유쟁점수, round(가중합,2) AS 점수, (t IS NOT NULL) AS 세목일치
ORDER BY 점수 DESC LIMIT 20;
// → "공유 2개+ & 세목 일치"가 심판↔판례 다리의 실용 정밀도 구간
```

### 사전 확장 방법 (수기 아님)

키워드는 **substantive 제목의 빈출 토큰에서 자동 추출**한다. 세목명
(TaxItem 기보유)·문서 정형구(청구인·청구주장)·과광범위어(부동산)는 제외:

```cypher
CALL apoc.periodic.iterate(
  "MATCH (cn:CaseName) WHERE cn.issue_class = 'substantive' RETURN cn",
  "SET cn.norm_text = apoc.text.replace(cn.name,
       '\\s*\\((각하|기각|경정|취소|인용|재조사|일부인용|일부기각)\\)\\s*$', '')",
  {batchSize: 10000});

MATCH (t:TaxItem) WITH collect(t.name) AS 세목들
MATCH (cn:CaseName) WHERE cn.norm_text IS NOT NULL
UNWIND split(cn.norm_text, ' ') AS raw
WITH 세목들, apoc.text.replace(
       apoc.text.replace(raw, '[^가-힣0-9]', ''),
       '(을|를|이|가|의|에|은|는|로|으로|와|과|도|만|에서|에게)$', '') AS token
WHERE size(token) >= 3 AND NOT token IN 세목들
  AND NOT token IN ['여부','당부','해당','처분','경우','것인지','인지','대하여',
                    '있는지','과세한','한다는','의하여','정당한','적용하여','아니한',
                    '청구인','청구주장','청구법인','부동산']
WITH token, count(*) AS freq WHERE freq >= 100
RETURN token, freq ORDER BY freq DESC LIMIT 80;
```

### (구) 문자열 클러스터링 절차 — 1층으로 유지

**용어 구분**: `CaseName`(제목)은 **노드**이며
`(d:DocNumber)-[:NAMED_AS]->(cn:CaseName)`로 연결(프로퍼티 아님.
entity_ruler의 `case_name` 파라미터는 사건번호를 뜻하는 동명이의).

- **CaseName(제목)** = "쟁점토지를 이 건 건축물의 부속토지로 보아…의 당부"
  — 사건 고유의 구체적 질문 문장
- **Issue(쟁점)** = "부속토지 판정" — 여러 문서를 묶는 추상 쟁점

실측: 제목 234,469개 중 **공유 27,393(11.7%)** — 이미 부분적 다리 존재
(`precedent_check` op이 CaseName을 앵커로 사용 중). 최대공유 1,254는 이미
Issue급으로 일반화된 표제. **과제 = 단독 88%를 정규화해 기존 다리에 합류.**

```cypher
// (1) 정규화 규칙 근거 확보 — 상위 공유 제목의 형태 확인 (규칙 조정용)
MATCH (d:DocNumber)-[:NAMED_AS]->(cn:CaseName)
WITH cn, count(d) AS 문서수 WHERE 문서수 >= 100
RETURN cn.name AS 제목, 문서수 ORDER BY 문서수 DESC LIMIT 20;

// (2) 정규화 키 부여 — 사건 고유 요소 제거, 쟁점 골자만
CALL apoc.periodic.iterate(
  "MATCH (cn:CaseName) RETURN cn",
  "SET cn.issue_key = cn.name
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '이 ?건|쟁점|당부|여부|해당하는지|청구주장', '')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '구\\\\s+', '')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '\\\\(\\\\d{4}[^)]*개정[^)]*\\\\)', '')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '제(\\\\d+)\\\\s*조의?(\\\\d*)', '$1조$2')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '제\\\\d+\\\\s*[항호목]', '')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '[\\\\d,]+원|[\\\\d,.]+㎡', '')
   SET cn.issue_key = apoc.text.replace(cn.issue_key, '\\\\d{4}\\\\.\\\\s?\\\\d{1,2}\\\\.\\\\s?\\\\d{1,2}\\\\.?', '')
   SET cn.issue_key = trim(apoc.text.replace(cn.issue_key, '\\\\s+', ' '))",
  {batchSize: 10000});

// (3) 정규화 효과 계측 — 공유 증가분 + 유형 교차 (승격 전 필수 확인)
MATCH (d:DocNumber)-[:NAMED_AS]->(cn:CaseName)
WHERE cn.issue_key IS NOT NULL AND cn.issue_key <> ''
WITH cn.issue_key AS k, count(DISTINCT d) AS n,
     collect(DISTINCT CASE WHEN d.name STARTS WITH '조심' OR d.name STARTS WITH '국심'
             THEN '심판' WHEN d:Directive THEN '예규' ELSE '판례' END) AS 유형들
WHERE n >= 3
RETURN count(k) AS 후보쟁점수,
       count(CASE WHEN size(유형들) >= 2 THEN 1 END) AS 유형교차쟁점수;

// (4) Issue 승격 + HAS_ISSUE
MATCH (d:DocNumber)-[:NAMED_AS]->(cn:CaseName)
WHERE cn.issue_key IS NOT NULL AND cn.issue_key <> ''
WITH cn.issue_key AS issue_key,
     collect(DISTINCT d) AS docs, collect(DISTINCT cn.name)[0..10] AS aliases
WHERE size(docs) >= 3
MERGE (i:Issue {name: issue_key})
  ON CREATE SET i.source = 'casename', i.aliases = aliases
SET i.support = size(docs)
WITH i, docs UNWIND docs AS d
MERGE (d)-[h:HAS_ISSUE]->(i)
  ON CREATE SET h.source = 'casename', h.confidence = 1.0;

// (5) 쟁점 사전 매칭 — summary 스캔으로 커버리지 확장 (증분 안전, 멱등)
:param taxonomy => [
  {issue: '1세대 1주택 비과세', aliases: ['1세대 1주택','1세대1주택','일시적 2주택']},
  {issue: '8년 자경 감면',      aliases: ['8년 자경','자경농지','재촌자경']},
  {issue: '부당행위계산부인',   aliases: ['부당행위계산']},
  {issue: '명의신탁 증여의제',  aliases: ['명의신탁']},
  {issue: '부속토지 판정',      aliases: ['부속토지','건축물의 효용','진입도로']}
];
CALL apoc.periodic.iterate(
  "UNWIND $taxonomy AS t
   MATCH (d:DocNumber) WHERE d.summary IS NOT NULL
     AND any(a IN t.aliases WHERE d.summary CONTAINS a)
   RETURN d, t",
  "MERGE (i:Issue {name: t.issue})
     ON CREATE SET i.source = 'dict', i.aliases = t.aliases
   MERGE (d)-[h:HAS_ISSUE]->(i)
     ON CREATE SET h.source = 'dict', h.confidence = 0.9",
  {batchSize: 2000, params: {taxonomy: $taxonomy}});

// (6) CO_OCCURS 파생 — 쟁점 지도
MATCH (i1:Issue)<-[:HAS_ISSUE]-(d:DocNumber)-[:HAS_ISSUE]->(i2:Issue)
WHERE elementId(i1) < elementId(i2)
WITH i1, i2, count(d) AS c WHERE c >= 5
MERGE (i1)-[r:CO_OCCURS]->(i2) SET r.count = c;
```

## 2-5. 【신규】 결론 상반 인용 — `diverges` 플래그 (법리 변경·구별 탐지)

`reference_date`(정확한 선후) + `canonical`(극성)이 동시에 확보되면서
가능해진 관계 신호. **선례를 인용했으면서 결론이 반대인 경우**를 표시한다
— 법리 변경(overruling), 사실관계 구별(distinguishing), 또는 판단 경향
변화의 후보.

**실물 사례** (업로드 2건):

| | 조심2020지1785 | 조심2023지0457 |
|---|---|---|
| 결정일 | 2021.06.14 | 2023.12.22 |
| 결정유형 | **취소**(납세자_승) | **기각**(납세자_패) |
| 쟁점 | 사업용 겸용 엘리베이터 → 고급주택 아님 | 엘리베이터 자체가 요건 → 고급주택 |
| 관계 | ← 0457이 1785를 **참조** | |

동일 쟁점에 정반대 판단. "이 선례 아직 유효한가?"에 답하려면 반드시
포착돼야 하는 신호이며, 현재 그래프에는 표현이 없다.

```cypher
// 인용 관계에 극성 상반 플래그 부여 (엣지 속성 — 신규 관계 타입 불요)
CALL apoc.periodic.iterate(
  "MATCH (citer:DocNumber)-[r:ADOPTS_INTENT]->(cited:DocNumber)
   WHERE coalesce(r.rel_kind, 'citation') = 'citation'    // 동시결정 형제 오탐 방지
   MATCH (citer)-[:RESULTED_IN]->(c1:DecisionType)
   MATCH (cited)-[:RESULTED_IN]->(c2:DecisionType)
   WHERE c1.canonical STARTS WITH '납세자' AND c2.canonical STARTS WITH '납세자'
   RETURN r, c1, c2",
  "WITH r, c1, c2,
        CASE WHEN c1.canonical IN ['납세자_승','납세자_일부승'] THEN '승' ELSE '패' END AS p1,
        CASE WHEN c2.canonical IN ['납세자_승','납세자_일부승'] THEN '승' ELSE '패' END AS p2
   SET r.diverges = (p1 <> p2),
       r.citer_canonical = c1.canonical, r.cited_canonical = c2.canonical",
  {batchSize: 5000});
// ※ left(canonical,5) 문자열 비교는 '일부승'(납세자_일)과 '승'(납세자_승)을
//   상반으로 오탐 — 반드시 계열 매핑으로 비교

// 규모 확인 — 상반 인용 비율
MATCH ()-[r:ADOPTS_INTENT]->() WHERE r.diverges IS NOT NULL
RETURN r.diverges, count(*) ORDER BY count(*) DESC;

// 활용 (a) — 이 선례를 인용한 후속 중 결론이 갈린 것 ("아직 유효한가")
MATCH (cited:DocNumber {name:$doc_number})<-[r:ADOPTS_INTENT]-(citer:DocNumber)
RETURN citer.name AS 후속결정, citer.reference_date AS 결정일,
       r.diverges AS 결론상반, r.citer_canonical AS 후속결과
ORDER BY 결정일 DESC;

// 활용 (b) — 법리 변경 후보: 상반 인용이 시간순으로 누적되는 선례
MATCH (cited:DocNumber)<-[r:ADOPTS_INTENT {diverges: true}]-(citer:DocNumber)
WITH cited, count(citer) AS 상반수, max(citer.reference_date) AS 최근상반
MATCH (cited)<-[:ADOPTS_INTENT]-(all_citer)
WITH cited, 상반수, 최근상반, count(all_citer) AS 총인용
WHERE 총인용 >= 3 AND 상반수 * 1.0 / 총인용 >= 0.5
RETURN cited.name AS 흔들리는선례, 총인용, 상반수, 최근상반
ORDER BY 상반수 DESC LIMIT 20;
```

**주의 — 상반이 곧 법리 변경은 아님.** 사실관계가 달라 결론이 갈리는
정상적 구별(distinguishing)이 다수일 것이므로, 이 플래그는 **후보 제시**
용도이며 답변에서는 "결론이 달랐다"는 사실만 전달하고 원인 단정은 금지
(코드측 직조 규칙에 반영 — C13).

**인용 시차(`day_gap`)와 결합**하면 "최근까지 인용되는 살아있는 선례"와
"오래전 인용만 있는 사문화 선례"도 구분 가능:

```cypher
MATCH (cited:DocNumber)<-[r:ADOPTS_INTENT]-(citer:DocNumber)
WHERE citer.reference_date IS NOT NULL
WITH cited, count(citer) AS 총인용, max(citer.reference_date) AS 최종인용일
WHERE 총인용 >= 5
RETURN cited.name AS 선례, 총인용, 최종인용일,
       CASE WHEN 최종인용일 >= '2022-01-01' THEN '활성' ELSE '휴면' END AS 상태
ORDER BY 총인용 DESC LIMIT 20;
```

## 2-6. 활용 쿼리

```cypher
// 동일 쟁점 선행 선례 (reg_date 또는 year 필요)
MATCH (d:DocNumber {name:$doc_number})-[:HAS_ISSUE]->(i:Issue)
MATCH (i)<-[:HAS_ISSUE]-(p:DocNumber)-[:RESULTED_IN]->(dt)
WHERE coalesce(p.ref_year, 9999) < coalesce(d.ref_year, 0) AND NOT p:Directive
RETURN i.name AS 쟁점, p.name AS 선행선례, dt.canonical AS 결과 LIMIT 15;

// 쟁점 승소율
MATCH (i:Issue) WHERE i.name CONTAINS $issue_keyword
MATCH (i)<-[:HAS_ISSUE]-(d:DocNumber)-[:RESULTED_IN]->(dt:DecisionType)
WHERE NOT coalesce(dt.canonical,'') IN
      ['__POLLUTED__','__CRIMINAL__','__PARTY_OTHER__','검색안됨']
RETURN i.name AS 쟁점, dt.canonical AS 결정유형, count(d) AS 건수
ORDER BY 쟁점, 건수 DESC;

// 결과 반전 쌍 — 비슷한데 결론이 반대 (SIMILAR_TO 기반, §5 참조)
MATCH (seed:DocNumber {name:$doc_number})-[:RESULTED_IN]->(sdt:DecisionType)
MATCH (seed)-[r:SIMILAR_TO]-(sim:DocNumber)-[:RESULTED_IN]->(dt:DecisionType)
WHERE NOT sim:Directive
  AND sdt.canonical STARTS WITH '납세자' AND dt.canonical STARTS WITH '납세자'
  AND (CASE WHEN sdt.canonical IN ['납세자_승','납세자_일부승'] THEN '승' ELSE '패' END)
   <> (CASE WHEN dt.canonical  IN ['납세자_승','납세자_일부승'] THEN '승' ELSE '패' END)
RETURN sim.name AS 반전사건, dt.canonical AS 상대결과, r.score AS 유사도
ORDER BY 유사도 DESC LIMIT 10;
```

---

# §3. 심판 ↔ 판례 관계 생성

**현황: 0건.** 참조/따른결정은 심판 내부 전용이라 교차가 없음. 신규 구축 대상.

## 3-1. 폐기된 접근 (재시도 금지)

| 접근 | 결과 |
|---|---|
| doc_metadata의 `prev_case`/`prev_lawsuit_case` | 5.10.1 실측 결과 **필드 부재**(5.10.2는 계획 기재) |
| 판례 summary에서 조심 번호 정규식 추출 | 실문서 4건 검증 — 요지에 **사건번호 없음**(쟁점 법리만 담는 포맷) |
| 판례 본문의 전심 서술 | "조세심판원에 심판을 청구하였으나 …기각" — **번호 없이 서술만** |

## 3-2. 축 C — CITES_PRECEDENT (상위 판례 인용, 정형 추출)

판결 이유에 상위 판례가 **정형으로** 인용됨("대법원 1994. 11. 11. 선고
94다28000 판결 참조"). 사건번호 추출이 실제로 되는 유일한 케이스.
`body_text`(이유 전문) 확보 시:

```cypher
CALL apoc.periodic.iterate(
  "MATCH (d:DocNumber) WHERE d.body_text IS NOT NULL
     AND d.body_text CONTAINS '선고' RETURN d",
  "UNWIND apoc.text.regexGroups(d.body_text,
     '(?:대법원|헌법재판소)\\\\s*\\\\d{2,4}\\\\s*[가-힣]{1,3}\\\\s*\\\\d{2,6}') AS g
   WITH d, apoc.text.replace(g[0], '\\\\s', '') AS cited
   MERGE (c:Precedent {name: cited})
   MERGE (d)-[r:CITES_PRECEDENT]->(c) ON CREATE SET r.source = 'body_regex'",
  {batchSize: 1000});

// 활용 — 같은 대법원 판례를 인용한 심판·판례 = 법리 형제 (유형 교차 다리)
MATCH (d1:DocNumber)-[:CITES_PRECEDENT]->(c:Precedent)<-[:CITES_PRECEDENT]-(d2:DocNumber)
WHERE elementId(d1) < elementId(d2)
RETURN c.name AS 공통상위판례, count(*) AS 형제쌍수 ORDER BY 형제쌍수 DESC LIMIT 20;

// 스텁이 코퍼스 내 실제 문서와 매칭되면 인용 관계로 흡수
MATCH (c:Precedent), (d:DocNumber {name: c.name})
MATCH (citer)-[:CITES_PRECEDENT]->(c)
MERGE (citer)-[:ADOPTS_INTENT]->(d)
WITH c DETACH DELETE c;
```

## 3-3. 축 D — APPEALED_TO (승계쌍: 심판 기각 → 행정소송)

같은 **분쟁**의 승계. 신호가 약하므로 2단 구성.

**summary 정의 확정**: 저장되는 summary는 **첫 장 메타카드**(문서번호·
결정유형·세목·귀속연도·제목·요지·내용·관련법령) 수준. 처분 세액은 둘째 장
이유에만 있어 summary에 없음 → 1차는 메타 조인, 2차가 세액 지문.

```cypher
// 선행 — 조인 키 보유 확인
MATCH (d:DocNumber)
RETURN (d.name STARTS WITH '조심' OR d.name STARTS WITH '국심') AS 심판여부,
       count(*) AS 문서, count(d.attribution_year) AS 귀속연도,
       count(d.defendant) AS 피고, count(d.body_text) AS 본문 LIMIT 2;

// (1차) 메타 다중 조인 — 세목 × 귀속연도 × 처분청 × 시간창
CALL apoc.periodic.iterate(
  "MATCH (t:DocNumber)
   WHERE (t.name STARTS WITH '조심' OR t.name STARTS WITH '국심')
     AND t.attribution_year IS NOT NULL RETURN t",
  "MATCH (p:DocNumber)
   WHERE NOT (p.name STARTS WITH '조심' OR p.name STARTS WITH '국심')
     AND NOT p:Directive
     AND p.attribution_year = t.attribution_year
     AND EXISTS { (t)-[:CONCERNS_TAX]->(:TaxItem)<-[:CONCERNS_TAX]-(p) }
     AND coalesce(p.ref_year, 9999) > coalesce(t.ref_year, 0)
     AND coalesce(p.ref_year, 9999) <= coalesce(t.ref_year, 0) + 2
   WITH t, p,
        (t.tax_office IS NOT NULL AND t.tax_office = p.defendant) AS office_match,
        size([ (t)-[:REFERS_TO]->(l)<-[:REFERS_TO]-(p) | l ]) AS shared_laws,
        EXISTS { (t)-[:RESULTED_IN]->(:DecisionType {canonical:'납세자_패'}) } AS lost
   WHERE office_match OR shared_laws >= 1
   MERGE (t)-[c:APPEALED_TO]->(p)
   SET c.source='meta_join', c.office_match=office_match, c.shared_laws=shared_laws,
       c.confidence =
         (CASE WHEN office_match THEN 0.6 WHEN shared_laws >= 2 THEN 0.4 ELSE 0.25 END)
         + (CASE WHEN lost THEN 0.2 ELSE 0.0 END)",
  {batchSize: 500});

// (2차) 세액 지문 승격 — body_text 확보 시. 세목+금액 인접만(거래·배당액 배제)
CALL apoc.periodic.iterate(
  "MATCH (d:DocNumber) WHERE d.body_text IS NOT NULL RETURN d",
  "SET d.amount_fp = apoc.coll.toSet([ x IN apoc.text.regexGroups(d.body_text,
     '(?:취득세|재산세|양도소득세|종합부동산세|지방교육세|법인세|부가가치세|상속세|증여세|종합소득세|가산세)\\\\s*([0-9]{1,3}(?:,[0-9]{3}){2,})\\\\s*원')
     | apoc.text.replace(x[1], ',', '')])",
  {batchSize: 5000});

MATCH (t:DocNumber)-[c:APPEALED_TO]->(p:DocNumber)
WITH t, c, p, [a IN coalesce(t.amount_fp,[]) WHERE a IN coalesce(p.amount_fp,[])] AS shared
WHERE size(shared) >= 2
SET c.shared_amounts=size(shared), c.source='amount_fp', c.confidence=0.95;

// 검증
MATCH ()-[c:APPEALED_TO]->() RETURN c.source, c.confidence, count(*)
ORDER BY count(*) DESC;
```

**극성은 hard filter가 아닌 가점** — canonical은 납세자 축이지만 ①원고가
납세자가 아닌 사건(`__PARTY_OTHER__`) ②인용 후 처분청 재과세로 소송화
③백필 오류 가능성 때문에 배제 대신 +0.2 가점.

**confidence 0.8+ 만** 답변 직조에 사용, 이하는 "관련 가능 사건" 약한 표현
+ 검수 대기.

## 3-4. 지방세판례 심급 체인

심급 헤더(`1심-전주지방법원 2018구합1814` / `2심-…` / `3심-…`)는
**지방세판례에만 존재**(국세 판례엔 없음). 파싱/ETL에서 헤더 쌍을 추출하면
적재는 단순 MERGE:

```cypher
// $pairs: [{lower:'2018구합1814', upper:'2019누1536'}, ...]
UNWIND $pairs AS p
MATCH (l:DocNumber {name: p.lower}), (u:DocNumber {name: p.upper})
MERGE (l)-[r:APPEALED_TO]->(u) SET r.source = 'instance_header', r.confidence = 1.0;
```

**선행 확인**: 통합 PDF(1~3심 한 파일)가 심급별 3개 DocNumber로 분리
적재되는지 — 대표 심급 1문서라면 문서 내부 정보라 적재 대상 아님.

## 3-5. 활용 쿼리

```cypher
// 이 심판이 소송으로 갔다면 결과는
MATCH (t:DocNumber {name:$doc_number})-[c:APPEALED_TO]->(p:DocNumber)
WHERE c.confidence >= 0.8
OPTIONAL MATCH (p)-[:RESULTED_IN]->(dt:DecisionType)
RETURN p.name AS 후속소송, dt.canonical AS 결과, c.confidence;

// 심급 관통 체인 (지방세판례)
MATCH path = (t:DocNumber {name:$doc_number})-[:APPEALED_TO*1..3]->(final)
WHERE NOT (final)-[:APPEALED_TO]->()
RETURN [n IN nodes(path) | n.name] AS 심급체인 ORDER BY length(path) DESC LIMIT 3;
```

---

# §4. 심판 ↔ 예규 관계 생성

**현황: 0건.** 예규는 사건번호 인용이 없고 관계법령만 정형(표본:
세정13407-1214 → 「지방세법」제40조). **법령 매개가 유일 채널.**

## 4-1. 법령 공유 다리 (기존 관계만으로 즉시)

```cypher
// 이 심판 쟁점과 관련된 현행 예규
MATCH (d:DocNumber {name:$doc_number})-[:REFERS_TO]->(l:Law)
MATCH (l)<-[:REFERS_TO]-(dir:DocNumber:Directive)
WHERE NOT (dir)-[:REVISED_TO]->()          // 현행만
RETURN dir.name AS 현행예규, collect(DISTINCT l.name) AS 매개법령,
       count(DISTINCT l) AS 공유법령수
ORDER BY 공유법령수 DESC LIMIT 10;
```

## 4-2. BRIDGES 물화 — 희소성 가중 (선택)

공유 조문이 희소할수록 다리 가치가 높다(IDF). 2홉을 1홉 가상 엣지로 물화:

```cypher
CALL apoc.periodic.iterate(
  "MATCH (t:DocNumber)-[:REFERS_TO]->(l:Law)<-[:REFERS_TO]-(p:DocNumber)
   WHERE (t.name STARTS WITH '조심' OR t.name STARTS WITH '국심')
     AND p:Directive
   WITH t, p, collect(l) AS shared_laws
   WHERE size(shared_laws) >= 2
   RETURN t, p, shared_laws",
  "WITH t, p, shared_laws,
        reduce(s = 0.0, l IN shared_laws |
               s + 1.0 / log(2 + count{ (l)<-[:REFERS_TO]-() })) AS idf
   MERGE (t)-[b:BRIDGES]->(p)
   SET b.shared_laws = size(shared_laws), b.idf_score = idf",
  {batchSize: 200});

// 분포 확인 후 컷오프 — 전량 물화 금지 (조합 폭발 위험)
MATCH ()-[b:BRIDGES]->() RETURN
  percentileCont(b.idf_score, 0.5) AS p50, percentileCont(b.idf_score, 0.9) AS p90,
  count(*) AS 전체;
```

## 4-3. 쟁점 매개 (§2-3 Issue 구축 후)

Issue가 예규까지 커버하면 법령 매개보다 정밀한 다리가 됨:

```cypher
// 쟁점 매개 유형 교차 — 심판 ↔ 현행 예규
MATCH (d:DocNumber {name:$doc_number})-[:HAS_ISSUE]->(i:Issue)
MATCH (i)<-[:HAS_ISSUE]-(x:DocNumber)
WHERE x <> d AND (NOT x:Directive OR NOT (x)-[:REVISED_TO]->())
RETURN i.name AS 쟁점,
       CASE WHEN x:Directive THEN '현행예규' ELSE '쟁송' END AS 유형,
       x.name AS 문서 ORDER BY 유형 LIMIT 20;
```

**전제 확인** — 예규에 CaseName/summary가 붙어 있어야 Issue 클러스터링 대상:

```cypher
MATCH (d:DocNumber) WHERE d:Directive
RETURN count(*) AS 예규, count{ (d)-[:NAMED_AS]->() } AS 제목연결,
       count(d.summary) AS 내용보유;
```

## 4-4. 예규 자체 체인 (기존)

```cypher
// 구예규 → 현행예규 → 다루는 법령
MATCH (old:DocNumber:Directive {name:$directive_name})-[:REVISED_TO*1..]->(new)
WHERE NOT (new)-[:REVISED_TO]->()
OPTIONAL MATCH (new)-[:REFERS_TO]->(l:Law)
RETURN new.name AS 현행예규, collect(DISTINCT l.name)[0..5] AS 다루는법령;
```

---

# §5. GDS

## 5-1. SIMILAR_TO 진단 (10분 — 재구축 전 필수)

```cypher
// ① 계보 오염율 — 인용쌍이 10%+ 면 후처리 삭제 필수
MATCH (a)-[r:SIMILAR_TO]->(b)
RETURN count(*) AS 전체,
       count(CASE WHEN (a)-[:ADOPTS_INTENT|EXAMINES_DECISION*1..3]-(b) THEN 1 END) AS 인용쌍;

// ② score 변별력 — stdev 작고 p10~p90 폭 좁으면 허브 지배
MATCH ()-[r:SIMILAR_TO]->()
RETURN percentileCont(r.score,0.1) AS p10, percentileCont(r.score,0.5) AS p50,
       percentileCont(r.score,0.9) AS p90, stdev(r.score) AS sd, count(r) AS n;

// ③ 메타-only 유사쌍 — 30%+ 면 재구축
MATCH (a)-[r:SIMILAR_TO]->(b)
WHERE NOT (a)-[:REFERS_TO]->(:Law)<-[:REFERS_TO]-(b)
  AND NOT (a)-[:ADOPTS_INTENT|EXAMINES_DECISION*1..2]-(b)
RETURN count(*) AS 메타only쌍, avg(r.score) AS 평균score;

// ④ 신선도 공백
MATCH (d:DocNumber) WHERE NOT d:Directive AND NOT (d)-[:SIMILAR_TO]-()
RETURN count(d) AS 무엣지문서, max(d.ref_year) AS 최신연도;
```

## 5-2. 재구축 (③ 판정 시) — 구조 전용

임베딩 주입(Qdrant→PCA)은 Cypher 불가로 보류. FastRP featureless로도
인용·법령 토폴로지 유사는 충분히 잡힘.

```cypher
CALL gds.graph.drop('doc_sim', false);

CALL gds.graph.project(
  'doc_sim',
  ['DocNumber', 'Law'],
  {
    REFERS_TO:     { orientation: 'UNDIRECTED' },
    ADOPTS_INTENT: { orientation: 'UNDIRECTED' }
  }
);
// 허브 라벨(TaxItem/DecisionType/CaseType/CaseName) 미포함이 핵심
// ※ EXAMINES_DECISION 미포함 — 둘 다 넣으면 겹치는 39,272쌍만 가중 2배
//    (이중 카운트). 대신 편측 따른결정 1,545건(약 3%)이 누락됨.
//    누락까지 없애려면 아래 Cypher 프로젝션으로 합집합+중복제거:
//
//    CALL gds.graph.project.cypher('doc_sim',
//      'MATCH (n) WHERE n:DocNumber OR n:Law RETURN id(n) AS id, labels(n) AS labels',
//      'MATCH (a:DocNumber)-[:ADOPTS_INTENT|EXAMINES_DECISION]-(b:DocNumber)
//       WITH DISTINCT a, b WHERE id(a) < id(b)
//       RETURN id(a) AS source, id(b) AS target, 1.0 AS w
//       UNION
//       MATCH (d:DocNumber)-[r:REFERS_TO]->(l:Law)
//       RETURN id(d) AS source, id(l) AS target, r.w AS w');
//    (DISTINCT 가 중복 제거, id(a)<id(b) 로 무방향 1회만 계수)
// ※ CaseName은 공유 11.7%로 준허브 — 포함 시 쟁점 유사가 반영되나
//    최대공유 1,254 노드가 지배할 위험. 별도 실험으로 비교 후 결정.

CALL gds.fastRP.mutate('doc_sim', {
  embeddingDimension: 256,
  iterationWeights: [0.8, 1.0, 1.0],
  mutateProperty: 'fastrp',
  randomSeed: 42
});

CALL gds.knn.write('doc_sim', {
  nodeLabels: ['DocNumber'],
  nodeProperties: ['fastrp'],
  topK: 10,
  similarityCutoff: 0.80,          // ② 분포 확인 후 조정
  concurrency: 4,                  // CE 상한
  writeConcurrency: 4,
  writeRelationshipType: 'SIMILAR_TO_V2',
  writeProperty: 'score'
});

// 후처리 — 인용쌍 제거 (topK 슬롯 회수)
MATCH (a)-[r:SIMILAR_TO_V2]->(b)
WHERE (a)-[:ADOPTS_INTENT|EXAMINES_DECISION*1..3]-(b)
DELETE r;
```

**검증 → 스왑** (진단 ①③ 재실행 + 대표 사건 5~10건 육안 대조 통과 후):

```cypher
:auto MATCH ()-[r:SIMILAR_TO]->()
      CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS;
:auto MATCH (a)-[r:SIMILAR_TO_V2]->(b)
      CALL { WITH a, r, b
        MERGE (a)-[n:SIMILAR_TO]->(b) SET n.score = r.score DELETE r
      } IN TRANSACTIONS OF 10000 ROWS;
```

## 5-3. PPR seed 확산 (HippoRAG 차용)

```cypher
CALL gds.graph.project('doc_comm', 'DocNumber',
  { SIMILAR_TO:    {orientation:'UNDIRECTED'},
    ADOPTS_INTENT: {orientation:'UNDIRECTED'} });

// 질의 시 — seed 사건들과 그래프상 맥락 권위가 높은 문서
MATCH (s:DocNumber) WHERE s.name IN $seed_names
WITH collect(s) AS seeds
CALL gds.pageRank.stream('doc_comm', { sourceNodes: seeds, dampingFactor: 0.85 })
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS 사건번호, score ORDER BY score DESC LIMIT 20;
```

## 5-4. importance_score — 피인용 기반 대체 검토

현행 ArticleRank 대신 §1-3(e)의 피인용 수가 더 신뢰도 높은 신호
(편집자 명시 기록, 검증 완료). 병행 비교:

```cypher
// 피인용 수를 프로퍼티로 적재 (citation만)
CALL apoc.periodic.iterate(
  "MATCH (l:DocNumber) RETURN l",
  "SET l.cited_count = count{ (l)-[r:EXAMINES_DECISION]->()
                              WHERE coalesce(r.rel_kind,'citation')='citation' }",
  {batchSize: 10000});

// 기존 importance_score 와 상위 20 비교 — 겹침이 낮으면 신호가 다름
MATCH (d:DocNumber) WHERE d.cited_count > 0
RETURN d.name, d.cited_count, d.importance_score
ORDER BY d.cited_count DESC LIMIT 20;
```

## 5-5. Louvain (선택)

```cypher
CALL gds.louvain.write('doc_comm', { writeProperty: 'community_id' });
// 활용: 검색 다양화(동일 커뮤니티 편중 억제), community×canonical 승소율
```

## 5-6. 상시 계측 (주간 크론)

```cypher
// RESOLVES_TO 커버리지 — '결정당시' 조회의 실질 recall
MATCH (l:Law)
RETURN EXISTS{ (l)-[:RESOLVES_TO]->() } AS 다리있음, count(*) AS 법령노드수;

MATCH (l:Law) WHERE NOT (l)-[:RESOLVES_TO]->()
RETURN l.law, count(*) AS 인용수 ORDER BY 인용수 DESC LIMIT 20;

// SIMILAR_TO 신선도
MATCH (d:DocNumber) WHERE NOT d:Directive AND NOT (d)-[:SIMILAR_TO]-()
RETURN count(d), max(d.ref_year);

// 관계별 총량 (빌드 간 대조용)
MATCH ()-[r]->() RETURN type(r) AS 관계, count(*) ORDER BY count(*) DESC;
```

---

# §6. 통계 분석 — p-value가 실제로 의미를 갖는 지점

**원칙: 귀무모형이 자명하지 않은 것만 검정한다.** 예컨대 "참조/따른결정이
역방향 대응하는가"는 z=67,819 (p≈10⁻⁹⁹⁸⁷⁶¹⁸⁷⁸)로 압도적 유의하지만,
편집자가 의도적으로 기입한 필드이므로 **유의한 게 당연해서 정보가 없다.**
아래는 결과를 사전에 알 수 없어 검정이 발견 도구로 기능하는 항목들.

## 6-1. 법령 공동인용 유의성 (실행 가능) — "법령의 날짜 외 의미"

두 조문이 **우연 이상으로 함께 인용**되는지. 유의하면 실무적 결합 조문
(실체조문+절차조문, 본법+시행령, 과세요건+감면특례 쌍 등)이며, 이는
날짜(시행본)와 무관한 **법령 간 의미 구조**다. 현행 BRIDGES의 IDF 가중
(휴리스틱)을 통계적 근거로 대체·보강할 수 있고, 군집화(§5)의 Law 축
가중과 law-law 엣지 신설의 근거가 된다.

귀무: 조문 A(nA건 인용)와 B(nB건 인용)가 독립 → 기대 공동인용 = nA·nB/N.
포아송 근사로 z = (obs − exp)/√exp.

```cypher
MATCH (d:DocNumber)-[:REFERS_TO]->(:Law) WITH count(DISTINCT d) AS N
MATCH (l1:Law)<-[:REFERS_TO]-(d:DocNumber)-[:REFERS_TO]->(l2:Law)
WHERE elementId(l1) < elementId(l2)
WITH N, l1, l2, count(DISTINCT d) AS obs
WHERE obs >= 5
WITH l1, l2, obs,
     count{ (l1)<-[:REFERS_TO]-() } * 1.0 * count{ (l2)<-[:REFERS_TO]-() } / N AS exp
WITH l1, l2, obs, exp, (obs - exp) / sqrt(exp) AS z
WHERE z > 5                            // 다중검정(조문쌍 수십만) 감안 보수 임계
RETURN l1.name AS 조문A, l2.name AS 조문B, obs AS 공동인용,
       round(exp,1) AS 기대, round(z,1) AS z
ORDER BY z DESC LIMIT 30;
```

활용: 상위 쌍을 `(l1)-[:CO_CITED {z, obs}]->(l2)`로 물화하면 법령 매개
탐색이 1홉이 되고, 예규↔심판 다리(§4)의 정밀도가 오른다.

## 6-2. 조문 개정 전후 판단 변화 (실행 가능) — 개입 분석

ax_ 개정일을 **개입 시점**으로 두고, 그 조문을 인용한 사건의 승패 극성이
전후로 유의하게 변했는지 2-비율 z 검정. 유의하면 "이 조문은 해당 개정으로
판단 지형이 바뀌었다"는 **조문별 실측 근거**가 생겨, 현행 프롬프트의
일반론적 "과거 사례 주의" 문구를 조문 단위 경고로 대체할 수 있다.
diverges(§2-5)의 법령 버전에 해당.

```cypher
// ※ ax_ 스키마 표기(AMENDED_TO·enforce_date)는 실 스키마에 맞춰 조정
MATCH (l:Law)-[:RESOLVES_TO]->(:ax_Article)-[:AMENDED_TO]->(a2:ax_Article)
WITH l, a2.enforce_date AS 개정일 WHERE 개정일 IS NOT NULL
MATCH (d:DocNumber)-[:REFERS_TO]->(l)
MATCH (d)-[:RESULTED_IN]->(dt:DecisionType)
WHERE d.reference_date IS NOT NULL
  AND dt.canonical IN ['납세자_승','납세자_일부승','납세자_패','납세자_일부패']
WITH l, 개정일,
     CASE WHEN date(d.reference_date) < date(개정일) THEN 'before' ELSE 'after' END AS 시기,
     CASE WHEN dt.canonical IN ['납세자_승','납세자_일부승'] THEN 1 ELSE 0 END AS 승
WITH l, 개정일, 시기, count(*) AS n, sum(승) AS w
WITH l, 개정일,
     sum(CASE WHEN 시기='before' THEN n END) AS n1, sum(CASE WHEN 시기='before' THEN w END) AS w1,
     sum(CASE WHEN 시기='after'  THEN n END) AS n2, sum(CASE WHEN 시기='after'  THEN w END) AS w2
WHERE n1 >= 20 AND n2 >= 20                      // 소표본 배제
WITH l, 개정일, n1, n2, 1.0*w1/n1 AS p1, 1.0*w2/n2 AS p2,
     1.0*(w1+w2)/(n1+n2) AS pool
WITH l, 개정일, n1, n2, p1, p2,
     (p2-p1) / sqrt(pool*(1-pool)*(1.0/n1 + 1.0/n2)) AS z
WHERE abs(z) > 3
RETURN l.name AS 조문, 개정일, n1 AS 개정전건수, round(p1,3) AS 개정전승소율,
       n2 AS 개정후건수, round(p2,3) AS 개정후승소율, round(z,2) AS z
ORDER BY abs(z) DESC LIMIT 20;
```

## 6-3. 쟁점·법령별 승소율 편차 (설계) — L2의 통계적 방어

"이 쟁점 승소율 70%"를 그대로 내보내면 n=7짜리 노이즈가 섞인다. 전체
기저율 대비 2-비율 z + **BH(FDR) 보정**으로 유의한 편차만 보고.
분할표는 Cypher, FDR 컷은 파이썬 한 줄. **세목 층화 필수**(6-7 참조).
활용: 승소율 op·QPG 직조가 "통계적으로 의미 있는 것만" 말하게 됨.

## 6-4. 연령 보정 피인용 (설계) — authority 공정화

현행 `후속적용수`(§1-3e)는 **노출 기간 교란**이 있다 — 1995년 결정은
30년, 2023년 결정은 2년치 인용 기회를 가졌다. 동일 결정연도 코호트 내
기대 인용 대비 포아송 초과로 보정하면 "젊은데 이미 많이 인용되는"
떠오르는 선례가 드러난다(실측 상위 20에 2023년 사건이 있었던 것이 단서).

```cypher
MATCH (l:DocNumber) WHERE l.ref_year IS NOT NULL AND l.cited_count IS NOT NULL
WITH l.ref_year AS 연도, avg(l.cited_count) AS 코호트평균, collect(l) AS docs
WHERE 코호트평균 > 0
UNWIND docs AS l
WITH l, 연도, 코호트평균, (l.cited_count - 코호트평균)/sqrt(코호트평균) AS z
WHERE l.cited_count >= 5 AND z > 3
RETURN l.name AS 선례, 연도, l.cited_count AS 실제피인용,
       round(코호트평균,2) AS 코호트기대, round(z,1) AS z
ORDER BY z DESC LIMIT 20;
```

## 6-5. diverges 비율의 유의성 (설계)

현행 "흔들리는 선례" 규칙(총인용≥3 & 상반≥50%)은 3건 중 2건이면 걸려
소표본 노이즈에 취약. 전역 diverges 기저율 대비 **이항검정 + FDR**로
교체하면 오탐이 줄어든다.

## 6-6. SIMILAR_TO 품질의 순열검정 (설계) — G1 게이트 정량화

유사쌍이 무작위쌍 대비 법령 공유·canonical 일치·쟁점 공유를 얼마나
초과하는지 순열검정(무작위 쌍 1만 개 샘플링과 비교). 현행 G1 진단 임계
("메타-only 30%+" 등)가 감각치였던 것을 원칙적 기준으로 대체.

## 6-7. 이 그래프 특유의 함정 (전 항목 공통)

| 함정 | 내용 | 대응 |
|---|---|---|
| **다중검정** | 조문쌍은 수십만 — z>2 수준은 가짜가 쏟아짐 | FDR(BH) 보정 또는 z>5 보수 임계 |
| **심슨의 역설** | 세목이 교란변수(양도세·취득세는 기저 승소율 자체가 다름) | 승소율 관련 검정은 **세목 층화** 후 |
| **노출 기간** | 피인용·공동인용 모두 오래된 문서가 유리 | 연도 코호트 내 비교(6-4) |
| **유의 ≠ 인과** | "개정 후 승소율 변화"가 개정 탓인지 사건 구성 변화인지 통계는 못 가름 | 직조에서는 "전후로 경향이 달라졌다"까지만, **원인 단정 금지**(C13과 동일) |

---

# §7. 고급 분석 기법 — 검증 후 반영 대상

각 항목은 **①실측 검증 → ②판정 기준 → ③반영 위치** 순으로 구성.
검증 없이 도입하지 않는다(§6-7의 함정 원칙과 동일).

## 7-1. 시간 존중 경로 (time-respecting path) — ⚠ 현재 버그, 즉시 수정

**문제**: 법리 뿌리 체인(§1-3c)이 `ADOPTS_INTENT*1..3` 을 순회하면서
**시간 단조성을 강제하지 않는다.** 데이터에 방향 위반이 실재하므로
(구 빌드 미래→과거 388건), "A(2020)→B(2015)→C(2018)" 같이 **시간을
거스르는 인용 체인**이 답변으로 나갈 수 있다. 시간 인지 그래프에서
경로는 시각 단조성을 만족해야 유효하다.

**① 검증 — 위반 경로 규모**
```cypher
MATCH path = (d:DocNumber)-[:ADOPTS_INTENT*2..3]->(root:DocNumber)
WHERE all(n IN nodes(path) WHERE n.reference_date IS NOT NULL)
WITH path,
     all(i IN range(0, length(path)-1)
         WHERE date(nodes(path)[i].reference_date)
             > date(nodes(path)[i+1].reference_date)) AS 시간정합
RETURN 시간정합, count(*) AS 경로수;
```

**② 판정**: 위반 경로가 존재하면(0이 아니면) 무조건 수정 대상.

**③ 반영 — 순회에 단조 제약 추가**
```cypher
MATCH path = (d:DocNumber {name:$doc_number})-[:ADOPTS_INTENT*1..3]->(root:DocNumber)
WHERE NOT (root)-[:ADOPTS_INTENT]->()
  AND all(r IN relationships(path) WHERE coalesce(r.rel_kind,'citation') = 'citation')
  AND all(i IN range(0, length(path)-1)
          WHERE date(nodes(path)[i].reference_date)
              > date(nodes(path)[i+1].reference_date))   // 엄격 단조 감소
RETURN [n IN nodes(path) | n.name] AS 법리전파체인, length(path) AS 깊이
ORDER BY 깊이 DESC LIMIT 5;
```
코드측: `adopts_intent_chain` op에 동일 제약 반영 (**C14**).

## 7-2. 경험적 베이즈 (Beta-Binomial) — 승소율 소표본 보정

**문제**: "자경농지 승소율 5/7 = 71%"를 그대로 LLM에 넘기면 소표본
노이즈가 사실처럼 전달된다. §6-3의 FDR은 "보고할지 말지"의 이진 결정인
반면, 경험적 베이즈는 **값 자체를 신뢰도에 맞게 축소(shrinkage)** 하므로
그대로 출력에 쓸 수 있다.

**① 데이터 추출**
```cypher
MATCH (i:Issue)<-[:HAS_ISSUE]-(d:DocNumber)-[:RESULTED_IN]->(dt:DecisionType)
WHERE dt.canonical IN ['납세자_승','납세자_일부승','납세자_패','납세자_일부패']
WITH i.name AS 쟁점, count(*) AS n,
     sum(CASE WHEN dt.canonical IN ['납세자_승','납세자_일부승'] THEN 1 ELSE 0 END) AS 승
RETURN 쟁점, n, 승, round(1.0*승/n, 3) AS 원시승소율 ORDER BY n DESC;
```

**② 적합·보정** (파이썬): 전역 (n, 승) 분포에 Beta(α,β) 적합 →
`보정승소율 = (승 + α) / (n + α + β)`. n이 작을수록 전역 평균으로 수축.

**③ 반영**: 승소율 op의 반환값을 보정치로 교체하고, `n`과 함께 노출
(LLM이 "표본 7건 기준"을 함께 서술하도록). 판정 기준 — 원시 대비 보정
편차가 큰 상위 쟁점을 표본 점검해 축소가 과한지 확인.

## 7-3. 생존분석 (Weibull / Kaplan-Meier) — 선례 수명

**동기**: §6-4의 연령 보정(코호트 평균 대비)은 **우편향 절단(censoring)을
무시**한다. 2023년 선례의 인용이 적은 것은 사문화가 아니라 관측 기간이
짧기 때문이며, 생존분석은 이를 원리적으로 처리한다.

- 사건(event): 선례의 **마지막 인용 이후 경과 시간**
- 절단(censored): 최근까지 인용된 선례
- **형상모수 k 해석**: k<1 → 감소 위험률(오래 인용된 선례가 계속 인용됨,
  누적우위) / k>1 → 노후화. 법리 인용망은 k<1 예상이나 **실측 필요**.

**① 이벤트 테이블 추출**
```cypher
MATCH (p:DocNumber)<-[:ADOPTS_INTENT]-(c:DocNumber)
WHERE p.reference_date IS NOT NULL AND c.reference_date IS NOT NULL
WITH p, max(date(c.reference_date)) AS 최종인용일, count(c) AS 총인용
RETURN p.name AS 선례, p.reference_date AS 선고일, 최종인용일, 총인용,
       duration.inDays(date(p.reference_date), 최종인용일).days AS 활동기간,
       duration.inDays(최종인용일, date('2026-01-01')).days AS 무인용기간;
```

**② 적합** (파이썬 `lifelines`): `무인용기간`을 duration, 최근 인용
여부를 event로 두고 Weibull 적합 → k, λ 추정.

**③ 반영**: `authority_ranking`·직조에 "사문화 확률" 제공.
`diverges`(§2-5)와 결합하면 **"이 선례 아직 유효한가"** 에 정면으로
답하는 신호가 된다. 판정 — k의 신뢰구간이 1을 포함하면 지수분포로
단순화(형상 정보 없음).

## 7-4. 변화점 탐지 (changepoint) — 개정일 가정의 검증

**동기**: §6-2는 **개정일을 변화 시점으로 가정**한다. 그러나 실무 경향은
개정 **이전에** 바뀔 수 있다(하급심 흐름이 먼저 형성되고 입법이 추인).
승소율 시계열에서 변화점을 **탐지**해 형식적 개정일과 대조하면:
일치 → 개정 효과 / 선행 → 실무가 먼저 변화 / 무관 → 개정 무영향.

**① 시계열 추출**
```cypher
MATCH (d:DocNumber)-[:REFERS_TO]->(l:Law {name: $law_name})
MATCH (d)-[:RESULTED_IN]->(dt:DecisionType)
WHERE d.reference_date IS NOT NULL
  AND dt.canonical IN ['납세자_승','납세자_일부승','납세자_패','납세자_일부패']
WITH toInteger(left(toString(d.reference_date),4)) AS 연도, dt
WITH 연도, count(*) AS n,
     sum(CASE WHEN dt.canonical IN ['납세자_승','납세자_일부승'] THEN 1 ELSE 0 END) AS 승
WHERE n >= 10
RETURN 연도, n, round(1.0*승/n,3) AS 승소율 ORDER BY 연도;
```

**② 탐지** (파이썬 `ruptures`, PELT): 승소율 시계열의 변화점 추정.

**③ 반영**: 개정일과 변화점의 관계를 조문별로 기록해, 프롬프트 경고를
"개정 시점" 대신 **실측 변화 시점** 기준으로. 주의 — 변화의 **원인**은
통계로 단정 불가(§6-7).

## 7-5. k-최단경로 (Yen's) — 다중 경로 설명

**동기**: "이 심판과 저 판례가 왜 관련되는가"에 단일 최단경로만 제시하면
설명이 빈약하다. **서로 다른 3개 경로**(법령 경유 / 쟁점 경유 / 인용
경유)를 병렬 제시하면 법률 추론에 맞는 설명이 된다 — 실무자는 연결
"여부"가 아니라 **어떤 논리로 연결되는지**를 본다.

```cypher
CALL gds.graph.project('path_g', ['DocNumber','Law','Issue'],
  { REFERS_TO:{orientation:'UNDIRECTED'}, HAS_ISSUE:{orientation:'UNDIRECTED'},
    ADOPTS_INTENT:{orientation:'UNDIRECTED'}, RELATED_CASE:{orientation:'UNDIRECTED'} });

MATCH (s:DocNumber {name:$from}), (t:DocNumber {name:$to})
CALL gds.shortestPath.yens.stream('path_g',
  { sourceNode: s, targetNode: t, k: 3 })
YIELD index, nodeIds, costs
RETURN index AS 경로번호,
       [id IN nodeIds | gds.util.asNode(id).name] AS 경로, costs;
```

**판정**: 반환된 3경로가 서로 다른 매개(법령/쟁점/인용)를 지나는지 확인.
전부 같은 유형 노드만 지나면 다중 경로의 설명 가치가 없으므로 미도입.

## 7-6. 검토 후 기각

| 기법 | 기각 사유 |
|---|---|
| **A\*** | 인용 그래프에 유효한 허용적 휴리스틱(목표까지 거리의 낙관적 추정)이 없음. 78,005 노드 규모는 Dijkstra/BFS로 충분해 이득 없음 |
| **벤포드 법칙** | 탈세 탐지는 본 시스템 목적 아님. 단 **파싱 QC**로는 한정 활용 가능(추출 세액 분포가 벤포드에서 크게 이탈하면 정규식 오류 의심) |
| **Steiner 트리** | "k개 사건을 최소 연결하는 부분그래프"는 NP-hard이며, 본 용도는 7-5의 k-최단경로로 충분 |

---

# §8. 인과추론 — 식별 가능한 것과 불가능한 것

**원칙**: 본 그래프에서 인과추론의 실무 가치는 **새 주장을 만드는 것보다
잘못된 주장을 차단하는 데** 있다. 8-5(식별 불가 목록)가 이 절의 핵심이며
LLM 직조 규칙으로 직결된다.

## 8-1. 인과 DAG — 교란 구조 문서화 (가장 실용적)

새 분석을 추가할 때마다 "무엇을 통제해야 하는가"를 이 표에서 확인한다.
§6-7의 "세목 층화 필수"를 원칙화한 것.

```
        세목 ─────────────→ 승소율        세목별 기저율 상이(양도세≠취득세)
         ↑                     ↑
        쟁점 ──────────────────┤          쟁점이 세목을 결정하기도 함
                               │
        연도 ──────────────────┤          시대 추세·법령 개정
       처분청 ─────────────────┤          지역·기관 차이
     사건강도(미관측) ─────────┘          ★ 관측 불가 = 근본 교란
```

| 분석 대상 | 필수 통제 | 비고 |
|---|---|---|
| 쟁점별 승소율 | 세목, 연도 | 세목 미통제 시 심슨 역설 |
| 조문 개정 효과 | 세목, 시대추세(대조군) | → 8-2 DiD |
| 처분청 효과 | 세목, 연도, 쟁점 | → 8-3 PSM |
| 선례 인용 효과 | **사건강도(관측 불가)** | → **식별 불가**(8-5) |

## 8-2. DiD (이중차분) — §6-2의 정직한 업그레이드

**문제**: §6-2의 단순 전후 비교는 **개정 효과와 시대 추세를 분리하지
못한다.** 2015→2020 승소율 상승이 개정 때문인지 전반적 추세인지 불명.

**해법**: 같은 기간 **개정되지 않은 유사 조문**(동일 세목)을 대조군으로
두고 추세를 차감:
`DiD = (처치_후 − 처치_전) − (대조_후 − 대조_전)`

**① 시계열 추출 (평행추세 검정 겸용)**
```cypher
:param law_treated => '소득세법 제89조';   // 개정 조문
:param law_control => '소득세법 제88조';   // 동일 세목·미개정
:param reform_date => date('2016-01-01');

MATCH (d:DocNumber)-[:REFERS_TO]->(l:Law)
WHERE l.name IN [$law_treated, $law_control]
MATCH (d)-[:RESULTED_IN]->(dt:DecisionType)
WHERE d.reference_date IS NOT NULL
  AND dt.canonical IN ['납세자_승','납세자_일부승','납세자_패','납세자_일부패']
WITH l.name AS 조문,
     toInteger(left(toString(d.reference_date),4)) AS 연도,
     CASE WHEN date(d.reference_date) < $reform_date THEN 'pre' ELSE 'post' END AS 시기,
     CASE WHEN dt.canonical IN ['납세자_승','납세자_일부승'] THEN 1 ELSE 0 END AS 승
WITH 조문, 시기, 연도, count(*) AS n, sum(승) AS w
WHERE n >= 10
RETURN 조문, 시기, 연도, n, round(1.0*w/n,3) AS 승소율
ORDER BY 조문, 연도;
```

**② 판정 — 두 관문을 모두 통과해야 유효**
- **평행추세**: 개정 **이전** 구간에서 처치·대조군 승소율 추이가 나란한가.
  어긋나면 DiD 무효(대조군 재선정).
- **플라시보**: 개정과 무관한 임의 시점을 가짜 개정일로 넣었을 때 효과가
  0에 가까운가. 유의한 효과가 나오면 설계 결함.

**③ 반영**: 통과 시에만 "이 조문은 X년 개정으로 판단 경향이 변했다"를
프롬프트 경고 근거로 사용. 미통과 시 §6-2의 단순 비교 수준으로 강등하고
"전후로 달라 보인다"까지만 서술.

## 8-3. 성향점수 매칭 — 처분청 효과

**질문**: "동종 사건인데 처분청이 다르면 결과가 달라지는가" — 지역 간
과세 일관성 이슈로, 실무·정책 가치가 크다.

**① 공변량 추출**
```cypher
MATCH (d:DocNumber)-[:RESULTED_IN]->(dt:DecisionType)
WHERE d.tax_office IS NOT NULL AND d.reference_date IS NOT NULL
  AND dt.canonical IN ['납세자_승','납세자_일부승','납세자_패','납세자_일부패']
OPTIONAL MATCH (d)-[:CONCERNS_TAX]->(t:TaxItem)
OPTIONAL MATCH (d)-[:HAS_ISSUE]->(i:Issue)
RETURN d.name, d.tax_office AS 처분청, t.name AS 세목,
       toInteger(left(toString(d.reference_date),4)) AS 연도,
       collect(DISTINCT i.name) AS 쟁점들,
       CASE WHEN dt.canonical IN ['납세자_승','납세자_일부승'] THEN 1 ELSE 0 END AS 승;
```

**② 매칭**(파이썬): 세목·연도·쟁점 집합으로 성향점수 추정 후 처분청 쌍별
매칭 → ATT 산출. **공통지지(common support) 확인 필수** — 특정 처분청에만
있는 쟁점 유형은 비교 불가이므로 제외.

**③ 반영**: 유의차 발견 시 내부 리포트용. **LLM 답변에는 노출 금지** —
"A청은 납세자에게 불리하다"는 서술은 오해·오용 소지가 크다.

## 8-4. RDD (회귀단절) — 조세법 문턱 (body_text 확보 시)

조세법에는 **날카로운 기준선**이 많고, 문턱 근처는 사실상 무작위 배정에
가까워 진짜 인과효과 식별이 가능하다(245㎡ vs 246㎡ 주택은 본질적으로
같으나 법적 취급만 다름).

| 문턱 | 기준 변수 |
|---|---|
| 고급주택 | 전용면적·가액 |
| 1세대1주택 비과세 | 보유·거주 기간 |
| 8년 자경농지 | 자경 기간 |
| 부속토지 | 배율(도시 5배 / 그 외 10배) |
| 청구기간 | 90일 |

**선행 조건**: 면적·기간·금액이 **구조화 필드로 추출**돼 있어야 함
(현재 body_text 내부에만 존재 — §3-3 세액 지문 추출과 동일한 선행조건).

**부수 활용 — bunching 분석**: 문턱 바로 아래에 사건이 과도하게 몰려
있으면 납세자의 전략적 회피 행동이 드러난다(조세경제학 표준 기법).

## 8-5. ★ 식별 불가 — LLM 금지 주장 목록

**이 절이 §8의 핵심.** 아래는 관측 데이터로 식별 **불가능**하므로,
상관을 인과로 서술하지 않도록 프롬프트 규칙에 반영한다(C13 계열).

| 금지 주장 | 왜 식별 불가 | 허용되는 서술 |
|---|---|---|
| **"이 선례를 인용하면 승소 가능성이 높다"** | 사건 강도(case merit)가 미관측 교란 — 강한 사건일수록 선례를 많이 인용하고 동시에 승소도 많이 함. 성향점수로도 통제 불가(매칭 변수에 merit 부재) | "이 선례를 인용한 사건 중 X건이 인용 결정되었다"(빈도 서술) |
| **"심판보다 소송 승소율이 높다/낮다"** | **선택편향** — 판례에는 심판에서 패소한 사건만 올라감. 모집단이 달라 비교 자체가 무의미 | 각 단계별 분포를 별도 제시, 비교 금지 |
| **"이런 사건은 이길 확률이 X%"** | **청구를 제기한 사건만 관측**. 승산 없다고 판단해 포기한 사건은 데이터에 부재 → 모든 승소율은 "청구를 결심한 조건부" | "청구가 제기된 사건 중에서는 X%"(조건 명시) |
| **"A 처분청은 납세자에게 불리하다"** | 8-3에서 유의차가 나와도 사건 구성 차이일 수 있고, 오용 위험이 큼 | 노출 금지(내부 리포트 한정) |
| **"이 쟁점은 최근 납세자에게 유리해졌다"** | 시대 추세·사건 구성 변화와 분리 불가(8-2 미통과 시) | "최근 N년 결정 중에서는 …"(기간 명시 빈도) |

**코드 반영 위치**: `prompts/templates/rag_system_prompt.j2` — 기존
"원인 단정 금지"(diverges) 규칙과 함께 배치. 승소율·통계 수치를 낼 때
**항상 표본 수(n)와 조건을 병기**하도록 지시.

---

# §9. 실행 순서

| 순서 | 작업 | 소요 | 수용 기준 |
|---|---|---|---|
| ⓪ | §0 canonical 백필 | 30분 | 잔여 NULL 0 + 분포 상식 부합 |
| ⓪-b | ✅ self-loop 제거 완료(38건) + ETL 자기참조 필터 추가 | 10분 | 완료 — B-1 재측정으로 판정 불변 확인 |
| ⓪-c | **§0-B 데이터 품질** — 유령 노드(연도 오타)·자릿수 변이 점검 | 30분 | 스텁 규모 파악 → 재연결/표시. **군집화·PageRank 전 필수** |
| ① | **reference_date 커버리지 확인 + ref_year 백필** + §5-6 계측 | 30분 | 결정일보유율 확인 → 시간 조건 쿼리 작동 |
| ② | ✅ B-1 교차검증 완료(임계 7일 확정) → **(C) rel_kind 확정 실행** | 10분 | merged 3,197건 분류 확인 |
| ③ | ✅ **§2-1 완료 + 성격 규명** — 보강 1,134(80%가 3년내 = 통상 누락) | 완료 | 잔여 citation 편측 **0** 확인. 원문 표본 대조만 선택적 |
| ④ | ✅ **§2-2 완료** — RELATED_CASE 2,405 무방향 | 완료 | 세목 순도 100%·차수 최대 26 검증됨 |
| ④-b | **§2-5 diverges 플래그** (결론 상반 인용) | 20분 | 상반 비율 + 흔들리는 선례 목록 |
| ⑤ | ✅ **§2-3 축 A 확정** — 키워드 20개 3유형 관통 | 완료 | 문자열 8.7% vs 키워드 100% 교차 |
| ⑥ | §2-3 (4)(5)(6) Issue 승격 | 반나절 | 유형별 HAS_ISSUE 커버리지 |
| ⑦ | §5-1 진단 → 판정 시 §5-2 재구축 | 10분 / 반나절 | 진단 ①③ 개선 + 육안 대조 |
| ⑧ | §3-3 APPEALED_TO 1차 (메타 조인) | 1~2h | confidence 분포 + 표본 검수 |
| ⑨ | §4-2 BRIDGES (선택) | 1h | idf 분포 확인 후 컷오프 |
| ⑩ | §5-3 프로젝션 · §5-4 피인용 비교 | 1h | PPR 샘플 · 상위 20 겹침률 |

| ⑪ | **§6-1 법령 공동인용 유의성** (z>5) | 1h | 상위 조문쌍 목록 → CO_CITED 물화 판단 |
| ⑫ | **§6-2 개정 전후 판단 변화** (n≥20, \|z\|>3) | 1h | 유의 조문 목록 → 프롬프트 경고 근거 |
| ⑬ | §6-4 연령 보정 피인용 | 30분 | authority 순위 재산출 비교 |

| ⑭ | **§7-1 시간 존중 경로** — 위반 규모 확인 → 순회 제약 추가 | 30분 | 위반 0 확인. 코드 C14 반영 |
| ⑮ | §7-2 경험적 베이즈 승소율 보정 | 반나절 | 원시 대비 보정 편차 점검 |
| ⑯ | §7-3 생존분석(Weibull) 선례 수명 | 1일 | k 신뢰구간이 1 제외하는지 |
| ⑰ | §7-4 변화점 탐지 vs 개정일 대조 | 1일 | 조문별 일치/선행/무관 분류 |
| ⑱ | §7-5 k-최단경로 다중 설명 | 2h | 3경로가 서로 다른 매개 경유하는지 |

| ⑲ | **§8-5 금지 주장 규칙 프롬프트 반영** | 30분 | 선례인용→승소 등 5개 금지 서술 차단 |
| ⑳ | §8-2 DiD (평행추세·플라시보 검정 포함) | 1일 | 두 관문 통과 시에만 개정 효과 서술 |
| ㉑ | §8-3 성향점수 매칭(처분청) | 1일 | 내부 리포트 한정, LLM 미노출 |

**body_text 확보 시 추가**: §3-2 CITES_PRECEDENT, §3-3 2차 세액 지문, **§8-4 RDD**(면적·기간·금액 구조화 필요).
**ETL 협업 필요**: §3-4 지방세판례 심급 헤더 추출.
