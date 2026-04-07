import { useState, useEffect, useRef, useCallback } from "react";

// ─── Design Tokens ───
const COLORS = {
  bg: "#F7F8FA",
  surface: "#FFFFFF",
  surfaceHover: "#F0F2F5",
  primary: "#1B64DA",
  primaryLight: "#E8F0FE",
  primaryDark: "#0D47A1",
  accent: "#00897B",
  accentLight: "#E0F2F1",
  warning: "#F57C00",
  warningLight: "#FFF3E0",
  danger: "#D32F2F",
  dangerLight: "#FFEBEE",
  success: "#2E7D32",
  successLight: "#E8F5E9",
  text: "#1A1D23",
  textSecondary: "#5F6368",
  textTertiary: "#9AA0A6",
  border: "#E1E3E6",
  borderLight: "#F0F1F3",
  chatBubbleAI: "#F0F4FA",
  chatBubbleUser: "#1B64DA",
};

const FONT = `'Pretendard Variable', 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif`;

// ─── Mock Data ───
const MOCK_NOTIFICATIONS = [
  { id: 1, type: "urgent", title: "종합소득세 신고 마감 D-7", desc: "5월 31일까지 신고·납부하셔야 합니다.", icon: "🏛️", date: "2026-05-24", category: "세금" },
  { id: 2, type: "benefit", title: "청년 월세 지원금 신청 가능", desc: "월 최대 20만원, 자격요건 충족 확인됨", icon: "🏠", date: "2026-04-07", category: "복지" },
  { id: 3, type: "info", title: "건강검진 대상자 안내", desc: "2026년 일반건강검진 대상입니다.", icon: "🏥", date: "2026-04-05", category: "의료" },
  { id: 4, type: "benefit", title: "국민취업지원제도 2유형 안내", desc: "구직촉진수당 월 50만원 지원 가능", icon: "💼", date: "2026-04-03", category: "고용" },
];

const MOCK_BENEFITS = [
  { id: 1, name: "청년 월세 특별지원", amount: "월 20만원 (최대 12개월)", deadline: "2026-06-30", score: 95, ministry: "국토교통부", status: "신청가능", tags: ["주거", "청년"] },
  { id: 2, name: "청년내일저축계좌", amount: "월 10만원 매칭 (3년)", deadline: "2026-05-15", score: 88, ministry: "보건복지부", status: "신청가능", tags: ["저축", "청년"] },
  { id: 3, name: "국민취업지원제도 II유형", amount: "월 50만원 (6개월)", deadline: "상시", score: 82, ministry: "고용노동부", status: "신청가능", tags: ["취업", "구직"] },
  { id: 4, name: "청년 교통비 지원", amount: "연 12만원", deadline: "2026-12-31", score: 78, ministry: "행정안전부", status: "신청가능", tags: ["교통", "청년"] },
  { id: 5, name: "문화누리카드", amount: "연 13만원", deadline: "2026-11-30", score: 72, ministry: "문화체육관광부", status: "확인필요", tags: ["문화", "여가"] },
];

const MOCK_WALLET = [
  { id: 1, name: "주민등록등본", issued: "2026-03-15", expires: "2026-06-15", issuer: "행정안전부" },
  { id: 2, name: "건강보험자격득실확인서", issued: "2026-04-01", expires: "2026-07-01", issuer: "건강보험공단" },
  { id: 3, name: "납세증명서", issued: "2026-02-20", expires: "2026-05-20", issuer: "국세청" },
];

const QUICK_DOCS = [
  { name: "주민등록등본", icon: "📄" },
  { name: "건강보험자격", icon: "🏥" },
  { name: "납세증명서", icon: "📋" },
];

const SAMPLE_QUESTIONS = [
  "이사했는데 뭘 해야 하나요?",
  "아이 낳으면 지원금 있나요?",
  "여권 잃어버렸어요",
  "청년 대상 혜택 알려주세요",
  "건강검진 어떻게 받나요?",
];

// ─── Claude API Integration ───
async function askClaude(messages) {
  const systemPrompt = `당신은 대한민국 공공기관 대민용 AI 비서입니다. 이름은 "나라도우미"입니다.

역할:
- 국민의 행정 민원, 복지 혜택, 세금, 건강검진, 여권, 주민등록 등 모든 공공 서비스에 대해 안내합니다.
- 일상적인 언어로 질문해도 행정 맥락을 정확히 파악하여 답변합니다.
- 관련 법령이나 근거가 있다면 간략히 언급합니다.
- 복수의 관련 서비스가 있다면 논리적 순서로 단계별 안내합니다.

답변 규칙:
- 한국어로 답변합니다.
- 친절하고 명확하게, 하지만 간결하게 답변합니다.
- 구체적인 절차가 있다면 단계별로 안내합니다.
- 필요한 서류나 준비물이 있다면 함께 안내합니다.
- 온라인 처리 가능 여부를 안내합니다 (정부24, 주민센터 등).
- 신청 기한이나 주의사항이 있다면 반드시 언급합니다.
- 정보가 불확실하면 "정확한 내용은 해당 기관(기관명)에 확인하시기 바랍니다"라고 안내합니다.
- 질문이 모호하면 구체화를 위한 역질문을 합니다.

금지사항:
- 확인되지 않은 금액이나 자격요건을 단정적으로 말하지 않습니다.
- 법률 자문이나 의료 조언을 하지 않습니다.`;

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        system: systemPrompt,
        messages: messages,
      }),
    });
    const data = await response.json();
    const text = data.content?.map((b) => (b.type === "text" ? b.text : "")).join("") || "죄송합니다. 일시적인 오류가 발생했습니다.";
    return text;
  } catch (e) {
    console.error("Claude API error:", e);
    return "죄송합니다. 네트워크 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.";
  }
}

// ─── Sub-components ───

function NotificationBadge({ count }) {
  if (!count) return null;
  return (
    <span style={{
      position: "absolute", top: -4, right: -4,
      background: COLORS.danger, color: "#fff",
      fontSize: 10, fontWeight: 700, borderRadius: 10,
      minWidth: 18, height: 18, display: "flex", alignItems: "center", justifyContent: "center",
      padding: "0 4px", border: `2px solid ${COLORS.surface}`,
    }}>{count}</span>
  );
}

function TabBar({ activeTab, onTabChange, notifCount }) {
  const tabs = [
    { id: "home", label: "홈", icon: "🏠" },
    { id: "chat", label: "대화", icon: "💬" },
    { id: "benefits", label: "혜택", icon: "🎁" },
    { id: "wallet", label: "내 지갑", icon: "👛" },
    { id: "settings", label: "설정", icon: "⚙️" },
  ];
  return (
    <nav style={{
      display: "flex", justifyContent: "space-around", alignItems: "center",
      background: COLORS.surface, borderTop: `1px solid ${COLORS.border}`,
      padding: "6px 0 env(safe-area-inset-bottom, 8px)", position: "sticky", bottom: 0, zIndex: 50,
    }}>
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onTabChange(t.id)} style={{
          background: "none", border: "none", cursor: "pointer",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
          color: activeTab === t.id ? COLORS.primary : COLORS.textTertiary,
          fontFamily: FONT, fontSize: 10, fontWeight: activeTab === t.id ? 700 : 500,
          position: "relative", padding: "4px 12px", transition: "color .2s",
        }}>
          <span style={{ fontSize: 20, lineHeight: 1 }}>{t.icon}</span>
          {t.id === "home" && <NotificationBadge count={notifCount} />}
          {t.label}
        </button>
      ))}
    </nav>
  );
}

function Header({ title, accessibilityMode, onToggleAccessibility }) {
  return (
    <header style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "12px 16px", background: COLORS.surface,
      borderBottom: `1px solid ${COLORS.border}`, position: "sticky", top: 0, zIndex: 50,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 24 }}>🇰🇷</span>
        <span style={{ fontFamily: FONT, fontWeight: 800, fontSize: accessibilityMode ? 20 : 16, color: COLORS.text }}>
          {title || "나라도우미"}
        </span>
      </div>
      <button onClick={onToggleAccessibility} aria-label="간편 화면 전환" style={{
        background: accessibilityMode ? COLORS.primary : COLORS.surfaceHover,
        color: accessibilityMode ? "#fff" : COLORS.textSecondary,
        border: "none", borderRadius: 20, padding: "6px 12px", cursor: "pointer",
        fontFamily: FONT, fontSize: accessibilityMode ? 14 : 12, fontWeight: 600,
        transition: "all .2s",
      }}>
        {accessibilityMode ? "✓ 간편모드" : "간편 화면"}
      </button>
    </header>
  );
}

// ─── Home Screen (UX-01) ───
function HomeScreen({ accessibilityMode, onNavigate }) {
  const fontSize = accessibilityMode ? 16 : 13;
  const titleSize = accessibilityMode ? 20 : 16;

  return (
    <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16, overflowY: "auto", flex: 1 }}>
      {/* 맞춤형 알림 카드 (UX-01-01) */}
      <section>
        <h2 style={{ fontFamily: FONT, fontSize: titleSize, fontWeight: 800, color: COLORS.text, margin: "0 0 10px" }}>
          지금 나에게 필요한 것
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {MOCK_NOTIFICATIONS.slice(0, 3).map((n) => (
            <button key={n.id} onClick={() => onNavigate(n.type === "benefit" ? "benefits" : "chat")} style={{
              display: "flex", alignItems: "center", gap: 12,
              background: n.type === "urgent" ? COLORS.warningLight : n.type === "benefit" ? COLORS.primaryLight : COLORS.surface,
              border: `1px solid ${n.type === "urgent" ? "#FFE0B2" : n.type === "benefit" ? "#BBDEFB" : COLORS.border}`,
              borderRadius: 12, padding: accessibilityMode ? "16px" : "12px", cursor: "pointer",
              width: "100%", textAlign: "left", fontFamily: FONT, transition: "transform .15s",
            }}>
              <span style={{ fontSize: accessibilityMode ? 28 : 24 }}>{n.icon}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontWeight: 700, fontSize: fontSize, color: COLORS.text }}>{n.title}</span>
                  {n.type === "urgent" && (
                    <span style={{
                      background: COLORS.warning, color: "#fff", fontSize: 10, fontWeight: 700,
                      padding: "1px 6px", borderRadius: 4,
                    }}>긴급</span>
                  )}
                </div>
                <span style={{ fontSize: fontSize - 1, color: COLORS.textSecondary, display: "block", marginTop: 2 }}>
                  {n.desc}
                </span>
              </div>
              <span style={{ color: COLORS.textTertiary, fontSize: 18 }}>›</span>
            </button>
          ))}
        </div>
      </section>

      {/* 원클릭 재발급 (UX-01-02) */}
      <section>
        <h2 style={{ fontFamily: FONT, fontSize: titleSize, fontWeight: 800, color: COLORS.text, margin: "0 0 10px" }}>
          자주 쓰는 서류 바로발급
        </h2>
        <div style={{ display: "flex", gap: 8 }}>
          {QUICK_DOCS.map((d) => (
            <button key={d.name} onClick={() => onNavigate("chat")} style={{
              flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
              background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 12,
              padding: accessibilityMode ? "20px 8px" : "16px 8px", cursor: "pointer", fontFamily: FONT,
              transition: "background .15s",
            }}>
              <span style={{ fontSize: accessibilityMode ? 32 : 28 }}>{d.icon}</span>
              <span style={{ fontSize: fontSize - 1, fontWeight: 600, color: COLORS.text, textAlign: "center" }}>{d.name}</span>
            </button>
          ))}
        </div>
      </section>

      {/* 내 지갑 요약 (UX-01-03) */}
      <button onClick={() => onNavigate("wallet")} style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: `linear-gradient(135deg, ${COLORS.primary}, ${COLORS.primaryDark})`,
        borderRadius: 14, padding: accessibilityMode ? "20px" : "16px", border: "none",
        cursor: "pointer", fontFamily: FONT, width: "100%",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 28 }}>👛</span>
          <div style={{ textAlign: "left" }}>
            <div style={{ color: "rgba(255,255,255,.8)", fontSize: fontSize - 1 }}>내 지갑</div>
            <div style={{ color: "#fff", fontWeight: 700, fontSize: fontSize + 2 }}>보관 증명서 {MOCK_WALLET.length}건</div>
          </div>
        </div>
        <span style={{ color: "rgba(255,255,255,.7)", fontSize: 22 }}>›</span>
      </button>

      {/* 생애주기 진행 인디케이터 (UX-01-04) */}
      <section style={{
        background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 16,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <span style={{ fontFamily: FONT, fontWeight: 700, fontSize: fontSize, color: COLORS.text }}>
            생애주기: 청년·구직기
          </span>
          <span style={{ fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.accent, fontWeight: 600 }}>4/7 완료</span>
        </div>
        <div style={{ height: 8, background: COLORS.borderLight, borderRadius: 4, overflow: "hidden" }}>
          <div style={{
            width: "57%", height: "100%", borderRadius: 4,
            background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.primary})`,
            transition: "width .5s ease",
          }} />
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
          {["주민등록", "건강보험", "국민연금", "취업지원"].map((s, i) => (
            <span key={s} style={{
              fontSize: fontSize - 2, padding: "3px 8px", borderRadius: 6,
              background: i < 4 ? COLORS.successLight : COLORS.surfaceHover,
              color: i < 4 ? COLORS.success : COLORS.textTertiary,
              fontFamily: FONT, fontWeight: 500,
            }}>✓ {s}</span>
          ))}
          {["세금신고", "자격증등록", "예비군편성"].map((s) => (
            <span key={s} style={{
              fontSize: fontSize - 2, padding: "3px 8px", borderRadius: 6,
              background: COLORS.surfaceHover, color: COLORS.textTertiary,
              fontFamily: FONT, fontWeight: 500,
            }}>○ {s}</span>
          ))}
        </div>
      </section>

      {/* 자연어 검색 (UX-05-01) */}
      <button onClick={() => onNavigate("chat")} style={{
        display: "flex", alignItems: "center", gap: 10,
        background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 12,
        padding: accessibilityMode ? "16px" : "12px 14px", cursor: "pointer", fontFamily: FONT, width: "100%",
      }}>
        <span style={{ fontSize: 18, color: COLORS.textTertiary }}>🔍</span>
        <span style={{ color: COLORS.textTertiary, fontSize: fontSize }}>
          무엇이든 물어보세요... "이사했는데 뭘 해야 하나요?"
        </span>
      </button>
    </div>
  );
}

// ─── Chat Screen (UX-02, FR-03, FR-04) ───
function ChatScreen({ accessibilityMode }) {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "안녕하세요! 나라도우미입니다 🇰🇷\n\n행정 민원, 복지 혜택, 세금, 건강검진 등 궁금한 점을 편하게 물어보세요." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const fontSize = accessibilityMode ? 16 : 14;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return;
    setShowSuggestions(false);
    const userMsg = { role: "user", content: text.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    const apiMessages = newMessages.filter((m) => m.role !== "system").map((m) => ({ role: m.role, content: m.content }));
    const reply = await askClaude(apiMessages);
    setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    setLoading(false);
  }, [messages, loading]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  // Quick reply chips (UX-02-03)
  const quickReplies = messages.length <= 1 ? [] : ["더 자세히 알려주세요", "신청 방법은요?", "필요한 서류가 뭔가요?", "온라인으로 가능한가요?"];

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
      {/* Messages */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            animation: "fadeInUp .3s ease",
          }}>
            <div style={{
              maxWidth: "85%", padding: accessibilityMode ? "14px 16px" : "10px 14px",
              borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              background: m.role === "user" ? COLORS.chatBubbleUser : COLORS.chatBubbleAI,
              color: m.role === "user" ? "#fff" : COLORS.text,
              fontFamily: FONT, fontSize: fontSize, lineHeight: 1.6, whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}>
              {m.content}
              {/* XAI 근거 보기 버튼 (UX-02-02) */}
              {m.role === "assistant" && i > 0 && (
                <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${COLORS.borderLight}` }}>
                  <button style={{
                    background: "none", border: "none", color: COLORS.primary, cursor: "pointer",
                    fontFamily: FONT, fontSize: fontSize - 2, fontWeight: 600, padding: 0,
                  }}>📎 근거 보기</button>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div style={{
              background: COLORS.chatBubbleAI, borderRadius: 16, padding: "12px 18px",
              fontFamily: FONT, fontSize: fontSize, color: COLORS.textSecondary,
            }}>
              <span className="typing-dots">답변 작성 중</span>
            </div>
          </div>
        )}

        {/* Suggestions */}
        {showSuggestions && messages.length <= 1 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            <span style={{ fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.textTertiary, fontWeight: 500 }}>
              이런 질문을 해보세요
            </span>
            {SAMPLE_QUESTIONS.map((q) => (
              <button key={q} onClick={() => sendMessage(q)} style={{
                background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 20,
                padding: "8px 14px", fontFamily: FONT, fontSize: fontSize - 1, color: COLORS.text,
                cursor: "pointer", textAlign: "left", transition: "background .15s",
              }}>{q}</button>
            ))}
          </div>
        )}

        {/* Quick replies (UX-02-03) */}
        {!loading && !showSuggestions && quickReplies.length > 0 && (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 4 }}>
            {quickReplies.map((q) => (
              <button key={q} onClick={() => sendMessage(q)} style={{
                background: COLORS.primaryLight, border: `1px solid #BBDEFB`, borderRadius: 16,
                padding: "6px 12px", fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.primary,
                cursor: "pointer", fontWeight: 600, transition: "background .15s",
              }}>{q}</button>
            ))}
          </div>
        )}
      </div>

      {/* 상담사 이관 CTA (UX-02-05) */}
      {messages.length > 3 && (
        <div style={{
          padding: "6px 16px", background: COLORS.warningLight, textAlign: "center",
          fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.warning,
          borderTop: `1px solid #FFE0B2`,
        }}>
          AI로 해결이 어려우신가요? <button style={{
            background: "none", border: "none", color: COLORS.primary, fontWeight: 700,
            cursor: "pointer", fontFamily: FONT, fontSize: fontSize - 2, textDecoration: "underline",
          }}>담당자 연결 →</button>
        </div>
      )}

      {/* Input area (UX-02-01) */}
      <div style={{
        padding: "10px 12px", background: COLORS.surface, borderTop: `1px solid ${COLORS.border}`,
        display: "flex", alignItems: "flex-end", gap: 8,
      }}>
        {/* 이미지 첨부 (UX-02-04) */}
        <button aria-label="파일 첨부" style={{
          background: "none", border: "none", fontSize: 22, cursor: "pointer",
          color: COLORS.textTertiary, padding: 4, flexShrink: 0,
        }}>📎</button>
        <div style={{
          flex: 1, display: "flex", alignItems: "flex-end",
          background: COLORS.bg, borderRadius: 20, border: `1px solid ${COLORS.border}`,
          padding: "4px 4px 4px 14px",
        }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="무엇이든 물어보세요..."
            rows={1}
            style={{
              flex: 1, border: "none", background: "transparent", resize: "none",
              fontFamily: FONT, fontSize: fontSize, color: COLORS.text,
              lineHeight: 1.5, padding: "6px 0", outline: "none",
              maxHeight: 100, overflow: "auto",
            }}
          />
          {/* 음성 입력 (FR-05-01) */}
          <button aria-label="음성 입력" style={{
            background: "none", border: "none", fontSize: 20, cursor: "pointer",
            color: COLORS.textTertiary, padding: 6, flexShrink: 0,
          }}>🎤</button>
        </div>
        <button onClick={() => sendMessage(input)} disabled={!input.trim() || loading} style={{
          background: input.trim() && !loading ? COLORS.primary : COLORS.borderLight,
          border: "none", borderRadius: "50%", width: 38, height: 38, flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: input.trim() && !loading ? "pointer" : "default",
          transition: "background .2s", color: "#fff", fontSize: 16,
        }}>↑</button>
      </div>
    </div>
  );
}

// ─── Benefits Screen (UX-03, FR-02) ───
function BenefitsScreen({ accessibilityMode }) {
  const [activeFilter, setActiveFilter] = useState("all");
  const [expandedId, setExpandedId] = useState(null);
  const fontSize = accessibilityMode ? 16 : 13;
  const filters = [
    { id: "all", label: "전체" }, { id: "housing", label: "주거" },
    { id: "employment", label: "고용" }, { id: "culture", label: "문화" },
  ];

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      <h2 style={{ fontFamily: FONT, fontSize: accessibilityMode ? 22 : 18, fontWeight: 800, color: COLORS.text, margin: 0 }}>
        AI가 찾은 나의 혜택
      </h2>
      <p style={{ fontFamily: FONT, fontSize: fontSize - 1, color: COLORS.textSecondary, margin: 0 }}>
        회원님의 정보를 분석하여 수혜 가능한 혜택을 선제적으로 안내합니다.
      </p>

      {/* Filters */}
      <div style={{ display: "flex", gap: 6 }}>
        {filters.map((f) => (
          <button key={f.id} onClick={() => setActiveFilter(f.id)} style={{
            background: activeFilter === f.id ? COLORS.primary : COLORS.surface,
            color: activeFilter === f.id ? "#fff" : COLORS.textSecondary,
            border: `1px solid ${activeFilter === f.id ? COLORS.primary : COLORS.border}`,
            borderRadius: 20, padding: "6px 14px", fontFamily: FONT,
            fontSize: fontSize - 1, fontWeight: 600, cursor: "pointer", transition: "all .2s",
          }}>{f.label}</button>
        ))}
      </div>

      {/* Benefit Cards (UX-03-01) */}
      {MOCK_BENEFITS.map((b) => (
        <div key={b.id} style={{
          background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14,
          overflow: "hidden", transition: "box-shadow .2s",
        }}>
          <button onClick={() => setExpandedId(expandedId === b.id ? null : b.id)} style={{
            width: "100%", padding: accessibilityMode ? "18px 16px" : "14px 16px",
            background: "none", border: "none", cursor: "pointer", fontFamily: FONT, textAlign: "left",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                  <span style={{ fontWeight: 700, fontSize: fontSize + 1, color: COLORS.text }}>{b.name}</span>
                  <span style={{
                    background: b.status === "신청가능" ? COLORS.successLight : COLORS.warningLight,
                    color: b.status === "신청가능" ? COLORS.success : COLORS.warning,
                    fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 4,
                  }}>{b.status}</span>
                </div>
                <div style={{ fontSize: fontSize, color: COLORS.primary, fontWeight: 700, marginBottom: 4 }}>
                  {b.amount}
                </div>
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  <span style={{ fontSize: fontSize - 2, color: COLORS.textSecondary }}>{b.ministry}</span>
                  <span style={{ color: COLORS.borderLight }}>·</span>
                  <span style={{ fontSize: fontSize - 2, color: COLORS.textSecondary }}>마감 {b.deadline}</span>
                </div>
              </div>
              {/* 적합도 점수 */}
              <div style={{
                width: 44, height: 44, borderRadius: "50%",
                background: b.score >= 90 ? COLORS.successLight : b.score >= 80 ? COLORS.primaryLight : COLORS.surfaceHover,
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
              }}>
                <span style={{
                  fontFamily: FONT, fontSize: 14, fontWeight: 800,
                  color: b.score >= 90 ? COLORS.success : b.score >= 80 ? COLORS.primary : COLORS.textSecondary,
                }}>{b.score}</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {b.tags.map((t) => (
                <span key={t} style={{
                  fontSize: 10, color: COLORS.textTertiary, background: COLORS.bg,
                  padding: "2px 6px", borderRadius: 4,
                }}>#{t}</span>
              ))}
            </div>
          </button>

          {/* Expanded: 원스톱 플로우 (UX-03-02) */}
          {expandedId === b.id && (
            <div style={{
              padding: "0 16px 16px", borderTop: `1px solid ${COLORS.borderLight}`,
              display: "flex", flexDirection: "column", gap: 10, paddingTop: 12,
            }}>
              <div style={{ display: "flex", gap: 8 }}>
                {["①동의", "②확인", "③신청", "④완료"].map((step, i) => (
                  <div key={step} style={{
                    flex: 1, textAlign: "center", padding: "8px 0", borderRadius: 8,
                    background: i === 0 ? COLORS.primaryLight : COLORS.bg,
                    fontFamily: FONT, fontSize: fontSize - 2, fontWeight: 600,
                    color: i === 0 ? COLORS.primary : COLORS.textTertiary,
                  }}>{step}</div>
                ))}
              </div>
              <button style={{
                background: COLORS.primary, color: "#fff", border: "none", borderRadius: 10,
                padding: accessibilityMode ? "16px" : "12px", fontFamily: FONT,
                fontSize: fontSize, fontWeight: 700, cursor: "pointer",
                transition: "background .2s",
              }}>지금 신청하기 →</button>
            </div>
          )}
        </div>
      ))}

      {/* 놓친 혜택 타임라인 (UX-03-03) */}
      <div style={{
        background: COLORS.surfaceHover, borderRadius: 14, padding: 16,
        border: `1px dashed ${COLORS.border}`,
      }}>
        <span style={{ fontFamily: FONT, fontSize: fontSize, fontWeight: 700, color: COLORS.textSecondary }}>
          📅 놓친 혜택
        </span>
        <div style={{ marginTop: 8, fontFamily: FONT, fontSize: fontSize - 1, color: COLORS.textTertiary, lineHeight: 1.6 }}>
          • 2026.03 — 청년 디지털 배움터 (마감) — 재신청 불가<br />
          • 2026.02 — 겨울 에너지 바우처 (마감) — 차기 접수 11월
        </div>
      </div>
    </div>
  );
}

// ─── Wallet Screen (UX-04, FR-07) ───
function WalletScreen({ accessibilityMode }) {
  const fontSize = accessibilityMode ? 16 : 13;
  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      <h2 style={{ fontFamily: FONT, fontSize: accessibilityMode ? 22 : 18, fontWeight: 800, color: COLORS.text, margin: 0 }}>
        내 지갑
      </h2>
      <p style={{ fontFamily: FONT, fontSize: fontSize - 1, color: COLORS.textSecondary, margin: 0 }}>
        발급받은 증명서와 자격을 안전하게 보관합니다.
      </p>

      {/* 증명서 카드 (UX-04-01) */}
      {MOCK_WALLET.map((w) => {
        const isExpiringSoon = new Date(w.expires) < new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);
        return (
          <div key={w.id} style={{
            background: `linear-gradient(135deg, #1A237E, #283593)`,
            borderRadius: 14, padding: accessibilityMode ? "20px" : "16px",
            color: "#fff", position: "relative", overflow: "hidden",
          }}>
            <div style={{
              position: "absolute", top: -20, right: -20, width: 80, height: 80,
              borderRadius: "50%", background: "rgba(255,255,255,.06)",
            }} />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontSize: fontSize + 2, fontWeight: 700, fontFamily: FONT }}>{w.name}</div>
                <div style={{ fontSize: fontSize - 1, opacity: 0.7, marginTop: 4, fontFamily: FONT }}>
                  발급: {w.issued} | 유효: {w.expires}
                </div>
                <div style={{ fontSize: fontSize - 2, opacity: 0.5, marginTop: 2, fontFamily: FONT }}>
                  {w.issuer}
                </div>
              </div>
              {isExpiringSoon && (
                <span style={{
                  background: "rgba(255,152,0,.3)", color: "#FFE082", fontSize: 10,
                  fontWeight: 700, padding: "3px 8px", borderRadius: 6, fontFamily: FONT,
                }}>만료임박</span>
              )}
            </div>
            {/* 제출처 제어 + 삭제/전송 (UX-04-02, UX-04-04) */}
            <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
              <button style={{
                background: "rgba(255,255,255,.15)", border: "1px solid rgba(255,255,255,.2)",
                borderRadius: 8, padding: "6px 12px", color: "#fff", fontFamily: FONT,
                fontSize: fontSize - 2, cursor: "pointer", fontWeight: 600,
              }}>제출 이력</button>
              <button style={{
                background: "rgba(255,255,255,.15)", border: "1px solid rgba(255,255,255,.2)",
                borderRadius: 8, padding: "6px 12px", color: "#fff", fontFamily: FONT,
                fontSize: fontSize - 2, cursor: "pointer", fontWeight: 600,
              }}>기관 전송</button>
              <button style={{
                background: "rgba(255,255,255,.08)", border: "1px solid rgba(255,255,255,.15)",
                borderRadius: 8, padding: "6px 12px", color: "rgba(255,255,255,.6)", fontFamily: FONT,
                fontSize: fontSize - 2, cursor: "pointer", fontWeight: 600,
              }}>삭제 요청</button>
            </div>
          </div>
        );
      })}

      {/* 데이터 활용 현황 (UX-04-03) */}
      <section style={{
        background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 16,
      }}>
        <h3 style={{ fontFamily: FONT, fontSize: fontSize, fontWeight: 700, color: COLORS.text, margin: "0 0 10px" }}>
          📊 내 데이터 활용 현황
        </h3>
        {[
          { service: "청년 월세 지원 자격확인", date: "2026-04-05", purpose: "자격요건 검증" },
          { service: "건강보험공단 자격조회", date: "2026-04-01", purpose: "자격득실 확인" },
          { service: "국세청 소득확인", date: "2026-03-20", purpose: "소득분위 판정" },
        ].map((d, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "8px 0", borderBottom: i < 2 ? `1px solid ${COLORS.borderLight}` : "none",
          }}>
            <div>
              <div style={{ fontFamily: FONT, fontSize: fontSize - 1, fontWeight: 600, color: COLORS.text }}>{d.service}</div>
              <div style={{ fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.textTertiary }}>{d.purpose}</div>
            </div>
            <span style={{ fontFamily: FONT, fontSize: fontSize - 2, color: COLORS.textTertiary }}>{d.date}</span>
          </div>
        ))}
      </section>
    </div>
  );
}

// ─── Settings Screen ───
function SettingsScreen({ accessibilityMode }) {
  const fontSize = accessibilityMode ? 16 : 13;
  const [channels, setChannels] = useState({ app: true, kakao: true, sms: false });
  const [categories, setCategories] = useState({ tax: true, welfare: true, health: true, education: false });

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ fontFamily: FONT, fontSize: accessibilityMode ? 22 : 18, fontWeight: 800, color: COLORS.text, margin: 0 }}>
        설정
      </h2>

      {/* 알림 채널 설정 (UX-03-04) */}
      <section style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 16 }}>
        <h3 style={{ fontFamily: FONT, fontSize: fontSize + 1, fontWeight: 700, color: COLORS.text, margin: "0 0 12px" }}>
          📢 알림 채널 설정
        </h3>
        {[
          { key: "app", label: "앱 푸시", icon: "📱" },
          { key: "kakao", label: "카카오톡", icon: "💬" },
          { key: "sms", label: "SMS 문자", icon: "✉️" },
        ].map((ch) => (
          <div key={ch.key} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "10px 0", borderBottom: `1px solid ${COLORS.borderLight}`,
          }}>
            <span style={{ fontFamily: FONT, fontSize: fontSize, color: COLORS.text }}>
              {ch.icon} {ch.label}
            </span>
            <button onClick={() => setChannels((p) => ({ ...p, [ch.key]: !p[ch.key] }))} style={{
              width: 44, height: 24, borderRadius: 12, border: "none", cursor: "pointer",
              background: channels[ch.key] ? COLORS.primary : COLORS.borderLight,
              position: "relative", transition: "background .2s",
            }}>
              <div style={{
                width: 20, height: 20, borderRadius: "50%", background: "#fff",
                position: "absolute", top: 2,
                left: channels[ch.key] ? 22 : 2, transition: "left .2s",
                boxShadow: "0 1px 3px rgba(0,0,0,.2)",
              }} />
            </button>
          </div>
        ))}
      </section>

      {/* 카테고리별 수신 설정 */}
      <section style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, borderRadius: 14, padding: 16 }}>
        <h3 style={{ fontFamily: FONT, fontSize: fontSize + 1, fontWeight: 700, color: COLORS.text, margin: "0 0 12px" }}>
          📋 카테고리별 알림
        </h3>
        {[
          { key: "tax", label: "세금·납부", icon: "🏛️" },
          { key: "welfare", label: "복지·지원금", icon: "🎁" },
          { key: "health", label: "건강·의료", icon: "🏥" },
          { key: "education", label: "교육·학자금", icon: "📚" },
        ].map((cat) => (
          <div key={cat.key} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "10px 0", borderBottom: `1px solid ${COLORS.borderLight}`,
          }}>
            <span style={{ fontFamily: FONT, fontSize: fontSize, color: COLORS.text }}>
              {cat.icon} {cat.label}
            </span>
            <button onClick={() => setCategories((p) => ({ ...p, [cat.key]: !p[cat.key] }))} style={{
              width: 44, height: 24, borderRadius: 12, border: "none", cursor: "pointer",
              background: categories[cat.key] ? COLORS.primary : COLORS.borderLight,
              position: "relative", transition: "background .2s",
            }}>
              <div style={{
                width: 20, height: 20, borderRadius: "50%", background: "#fff",
                position: "absolute", top: 2,
                left: categories[cat.key] ? 22 : 2, transition: "left .2s",
                boxShadow: "0 1px 3px rgba(0,0,0,.2)",
              }} />
            </button>
          </div>
        ))}
      </section>

      {/* 접근성 정보 */}
      <section style={{ background: COLORS.accentLight, border: `1px solid #B2DFDB`, borderRadius: 14, padding: 16 }}>
        <h3 style={{ fontFamily: FONT, fontSize: fontSize + 1, fontWeight: 700, color: COLORS.accent, margin: "0 0 8px" }}>
          ♿ 접근성 안내
        </h3>
        <p style={{ fontFamily: FONT, fontSize: fontSize - 1, color: COLORS.textSecondary, margin: 0, lineHeight: 1.6 }}>
          간편 화면 모드를 활성화하면 큰 글씨, 단순 메뉴, 고대비 색상으로 전환됩니다.
          시각장애인 회원은 앱 실행 시 음성 AI가 자동 연결됩니다.
        </p>
      </section>

      <div style={{
        textAlign: "center", fontFamily: FONT, fontSize: fontSize - 2,
        color: COLORS.textTertiary, padding: "16px 0",
      }}>
        나라도우미 POC v0.1 · 대민용 AI 비서 프로토타입<br />
        본 시스템은 개인정보보호법·마이데이터 가이드라인을 준수합니다.
      </div>
    </div>
  );
}

// ─── Main App ───
export default function PublicAIAssistant() {
  const [activeTab, setActiveTab] = useState("home");
  const [accessibilityMode, setAccessibilityMode] = useState(false);

  const navigate = useCallback((tab) => setActiveTab(tab), []);

  return (
    <div style={{
      width: "100%", maxWidth: 440, margin: "0 auto", height: "100vh",
      display: "flex", flexDirection: "column", background: COLORS.bg,
      fontFamily: FONT, overflow: "hidden",
      boxShadow: "0 0 60px rgba(0,0,0,.08)",
    }}>
      <style>{`
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
        * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
        html, body { height: 100%; overflow: hidden; }
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .typing-dots::after {
          content: '';
          animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
          0% { content: ''; }
          25% { content: '.'; }
          50% { content: '..'; }
          75% { content: '...'; }
        }
        textarea::placeholder { color: ${COLORS.textTertiary}; }
        button:hover { opacity: 0.92; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 4px; }
      `}</style>

      <Header
        accessibilityMode={accessibilityMode}
        onToggleAccessibility={() => setAccessibilityMode((p) => !p)}
      />

      {activeTab === "home" && (
        <HomeScreen accessibilityMode={accessibilityMode} onNavigate={navigate} />
      )}
      {activeTab === "chat" && (
        <ChatScreen accessibilityMode={accessibilityMode} />
      )}
      {activeTab === "benefits" && (
        <BenefitsScreen accessibilityMode={accessibilityMode} />
      )}
      {activeTab === "wallet" && (
        <WalletScreen accessibilityMode={accessibilityMode} />
      )}
      {activeTab === "settings" && (
        <SettingsScreen accessibilityMode={accessibilityMode} />
      )}

      <TabBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        notifCount={MOCK_NOTIFICATIONS.length}
      />
    </div>
  );
}
