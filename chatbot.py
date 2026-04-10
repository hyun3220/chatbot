# 상단 임포트 부분 수정 (기존꺼 다 지우고 이걸로 바꿔주세요)
import streamlit as st
import os

# 환경 변수로 v1을 강제 고정
os.environ["GOOGLE_API_VERSION"] = "v1"

import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# 페이지 설정
st.set_page_config(page_title="CLIP Report 5.0 AI 챗봇", page_icon="🤖")

#st.markdown("<div id='top-anchor'></div>", unsafe_allow_html=True)

# CSS (크기 축소, 반투명 스크롤 버튼, 다크/라이트모드 충돌 해결, 푸터 제거 등 모두 포함)
st.markdown("""
    <style>
        .scroll-container {
            position: fixed; bottom: 80px; right: 15px; z-index: 1000;
            display: flex; flex-direction: column; gap: 8px;
        }
        .scroll-link {
            width: 32px; height: 32px; background-color: #4F4F4F; color: white !important;
            border-radius: 50%; text-decoration: none !important; display: flex; 
            align-items: center; justify-content: center; font-size: 14px; 
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2); transition: all 0.3s ease; opacity: 0.3;
        }
        .scroll-link:hover {
            background-color: #000000; transform: scale(1.1); opacity: 1;
        }
        html { scroll-behavior: smooth; }
        html, body, [class*="css"] { font-size: 14px !important; line-height: 1.5 !important; }
        .stMarkdown p, .stMarkdown li, .stChatInput textarea { font-size: 14px !important; }
        .stButton button p { font-size: 13px !important; }
        h1, h2, h3 { font-size: 1.2rem !important; }
        header, footer { visibility: hidden !important; display: none !important; }
        .block-container { padding: 1rem !important; }

        /* [개선] 테마 대응형 슬림 카드 선택기 */
        div[data-testid="stRadio"] {
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 10px 14px !important;
            border: 1px solid rgba(128, 128, 128, 0.1);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }

        /* 라디오 버튼 제목 */
        div[data-testid="stRadio"] > label {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: var(--text-color) !important;
            margin-bottom: 10px !important;
            opacity: 0.7;
            text-align: left;
        }

        /* 버튼 그룹 컨테이너 */
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 10px !important;
            display: flex;
            flex-direction: row;
        }

        /* 기본 라디오 원형 아이콘 숨기기 */
        div[data-testid="stRadio"] div[role="radiogroup"] [data-testid="stWidgetSelectionControl"] {
            display: none !important;
        }

        /* 개별 버튼(칩) 스타일 */
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: rgba(128, 128, 128, 0.05) !important;
            padding: 6px 15px !important;
            border-radius: 10px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            min-width: fit-content !important;
            white-space: nowrap !important;
            flex: 1;
            display: flex;
            justify-content: center;
            border: 1px solid transparent !important;
            cursor: pointer !important;
            margin: 0 !important;
        }

        /* [핵심] 선택된 버튼 스타일 - 샥 이동하는 효과 */
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background-color: var(--background-color) !important;
            border: 1.5px solid #f97316 !important; /* 주황색 강조 */
            box-shadow: 0 4px 10px rgba(249, 115, 22, 0.15) !important;
            color: #f97316 !important;
            font-weight: bold !important;
            transform: translateY(-1px);
        }

        /* 마우스 호버 효과 */
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background-color: rgba(128, 128, 128, 0.1) !important;
        }

        /* 텍스트 정렬 */
        div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            font-size: 0.85rem !important;
            margin: 0 !important;
            text-align: center;
        }
    </style>

""", unsafe_allow_html=True)

# ==========================================
# 사이드바 구성 추가
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📎 CLIP Report 5.0 AI</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 0px; opacity: 0.2;'>", unsafe_allow_html=True)
    st.header("⚙️ 설정 및 정보")
    st.write("CLIP Report 5.0 및 eForm 5.0 매뉴얼 기반 지능형 챗봇입니다.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 대화 기록 초기화"):
        st.session_state.messages = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]
        st.rerun()

# API 키 설정
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    st.error("서버에 API 키가 설정되지 않았습니다.")
    st.stop()

@st.cache_resource
def get_retriever(API_KEY):
    # 기존기억을 지우고 분류된 데이터를 저장하기 위해 새 버전(v5) 폴더 사용
    CHROMA_PERSIST_DIR = "./chroma_db_v5"
    
    # 1. 소스 정의 (API 전용 URL / 디자이너 전용 PDF)
    # 리포트(R5) 전용
    r5_sources = {
        "url": "https://technet.hancomins.com/board/api/R5/symbols/ReportView.html",
        "pdf": "클립리포트 v5.0 매뉴얼.pdf",
        "category": "report"
    }
    # 이폼(E5) 전용
    e5_sources = {
        "url": "https://technet.hancomins.com/board/api/E5/symbols/Report.html",
        "pdf": "클립이폼 v5.0 매뉴얼.pdf",
        "category": "eform"
    }

    retrievers = {}

    for mode, config in [("report", r5_sources), ("eform", e5_sources)]:
        try:
            category_docs = []
            
            # (1) URL 로드 (API 정보)
            st.sidebar.text(f"URL 로딩 중 ({mode}): {config['url']}")
            loader = WebBaseLoader(config['url'])
            loaded_url = loader.load()
            for d in loaded_url:
                d.metadata["category"] = config["category"]
                d.metadata["source_type"] = "api"
            category_docs.extend(loaded_url)
            
            # (2) PDF 로드 (디자이너 정보)
            pdf_path = config["pdf"]
            if os.path.exists(pdf_path):
                st.sidebar.text(f"PDF 로딩 중 ({mode}): {pdf_path}")
                pdf_loader = PyPDFLoader(pdf_path)
                loaded_pdf = pdf_loader.load()
                for d in loaded_pdf:
                    d.metadata["category"] = config["category"]
                    d.metadata["source_type"] = "designer"
                category_docs.extend(loaded_pdf)
            else:
                st.sidebar.warning(f"{mode}용 PDF 파일을 찾을 수 없습니다: {pdf_path}")

            # (3) 쪼개기 (API 설명이 잘리지 않도록 크기 조정)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(category_docs)
            
            # (4) 시스템 구축 (Chroma + BM25)
            # BM25
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 15
            
            # Chroma (컬렉션 이름을 모드로 구분하여 완전 분리)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                ),
                persist_directory=f"{CHROMA_PERSIST_DIR}_{mode}",
                collection_name=f"clip_{mode}_v5"
            )
            
            # (5) 하이브리드 엔진 조합
            def build_hybrid_engine(bm25, vector):
                def hybrid_search(query: str):
                    docs_bm25 = bm25.invoke(query)
                    docs_chroma = vector.as_retriever(search_kwargs={"k": 15}).invoke(query)
                    
                    merged = []
                    seen = set()
                    for d1, d2 in zip(docs_bm25 + [None]*15, docs_chroma + [None]*15):
                        if d1 and d1.page_content not in seen:
                            seen.add(d1.page_content)
                            merged.append(d1)
                        if d2 and d2.page_content not in seen:
                            seen.add(d2.page_content)
                            merged.append(d2)
                    return merged[:20]
                return RunnableLambda(hybrid_search)

            retrievers[mode] = build_hybrid_engine(bm25_retriever, vectorstore)

        except Exception as e:
            st.sidebar.error(f"{mode} 엔진 구축 실패: {str(e)}")
            retrievers[mode] = None

    return retrievers

# 답변 생성 로직
def generate_answer(api_key, retriever, query, mode):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=api_key, temperature=0,
        #model="gemini-3-flash", google_api_key=api_key, temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # 모드에 따른 명칭 설정
    product_name = "클립리포트(CLIP report v5.0)" if mode == "report" else "클립이폼(CLIP eForm v5.0)"

    system_prompt = (
        f"당신은 현재 **[{product_name}]** 기술 문서를 기반으로 답변하는 전문 엔지니어입니다.\n"
        f"대화 기록에 다른 제품(예: 리포트 또는 이폼)의 내용이 있더라도, 이번 질문에는 반드시 현재 모드인 **[{product_name}]**의 context 정보만 사용하세요.\n"
        "만약 context에 해당 기능에 대한 정보가 없다면, 다른 제품의 지식을 빌려 답변하지 말고 정중히 모른다고 답변하세요.\n\n"
        "🔍 [문맥 추론 및 매핑 규칙]:\n"
        "- 고객님이 '순서 바꾸기', '위치 변경'이라고 말하면: 'Order', 'Index', 'Sequence', 'Sort' 관련 함수를 검색하세요.\n"
        "- 고객님이 '저장 옵션', '내보내기 설정'이라고 말하면: 'Save', 'Export', 'Format' 관련 함수를 검색하세요.\n"
        "- 고객님이 '안 보여주기', '숨기기'라고 말하면: 'Visible', 'Display', 'Hide' 관련 함수를 검색하세요.\n\n"    
        "예를 들어, '저장 순서 바꾸기'라는 질문을 받으면, Context에서 'Save'와 'Order'가 포함된 `setReportSaveMenuOrder` 같은 함수를 찾아내어 설명해야 합니다.\n"
        "질문의 단어와 문서의 단어가 100% 일치하지 않더라도, **기능적 의도**가 같다면 그 함수를 정답으로 제시하세요.\n"
        "당신은 '클립리포트(CLIP report v5.0)'의 모든 기술 문서를 완벽히 숙지한 **기술 지원 엔지니어**입니다.\n"
        "고객님이 아주 짧거나 모호하게 질문하더라도, 당신은 질문의 의도를 정확히 파악하여 전문적인 답변을 제공해야 합니다.\n"
        "예를 들어, 고객님이 '스크롤 뷰 사용법 알려줘'라고 질문을 할 경우, 인피니티 스크롤 뷰 함수를 찾는구나 라는 상황을 인지하고 파악하여 API를 전달할 수 있어야 합니다.\n\n"
        "💡 [답변 원칙]:\n"
        "1. **의도 파악**: 질문이 짧다면(예: 'PDF 인쇄'), 사용자가 '웹 브라우저의 인쇄 기능을 쓰려는 것인지' 혹은 '서버에서 직접 PDF를 생성하려는 것인지' 맥락을 구분하여 가장 적절한 API를 추천하세요.\n"
        "2. **정확성**: 함수명(예: report.print)은 대소문자와 철자를 Context와 대조하여 100% 일치할 때만 답변하세요. 확실하지 않다면 '비슷한 기능을 가진 함수'임을 명시하세요.\n"
        "3. **친절한 가이드**: 단순히 함수만 띡 던지지 말고, '이 함수는 어떤 상황에서 주로 사용되는지' 한 줄 설명을 덧붙이세요.\n\n"
        "🚨 [매우 중요 - 답변 구성 순서]:\n"
        "고객님이 무엇을 물어보든 항상 아래 순서로 답변하세요.\n"
        "1️⃣ **기능 요약**: 질문한 기능에 대한 짧은 정의\n"
        "2️⃣ **핵심 API**: 정확한 함수명과 파라미터 설명\n"
        "3️⃣ **전체 예시 코드**: 제공된 [웹 뷰어 스크립트 템플릿]을 활용한 완성된 HTML 코드\n"
        "4️⃣ **주의 사항**: 해당 API 사용 시 개발자가 자주 실수하는 부분이나 팁\n\n"
        "--- [웹 뷰어 스크립트 템플릿] ---\n"
        "<!DOCTYPE html>\n<html>\n<head>\n<script type='text/javascript'>\n"
        "var report;\nfunction html2xml(divPath){{\n"
        "    var reportkey = \"<%=resultKey%>\";\n"
        "    report = createReport(\"./report_server.jsp\", reportkey, document.getElementById(divPath));\n"
        "    \n    // 👇 여기에 질문한 API 적용\n    \n    report.view();\n}}\n"
        "</script>\n</head>\n<body><div id='targetDiv1'></div></body>\n</html>\n"
        "--- [템플릿 끝] ---\n\n"
        "문서에 없는 내용은 절대 지어내지 마세요. 모르는 내용은 '매뉴얼에서 해당 내용을 찾을 수 없으니 공식 기술 지원팀에 문의해 주세요'라고 안내하세요.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain.invoke(query)

# UI 및 실행 로직
st.title("")

# [수정] 테마 대응형 슬림 선택기
search_mode = st.radio(
    "어떤 제품에 대해 궁금하신가요?", 
    ["리포트(R5)", "이폼(E5)"], 
    index=0,
    horizontal=True
)

# 세션 상태 저장
st.session_state.search_mode = "report" if "리포트" in search_mode else "eform"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "무엇을 도와드릴까요? (API 및 디자이너 관련 문의만 가능합니다.)"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # 물리적으로 분리된 리트리버 딕셔너리 가져오기
        retriever_systems = get_retriever(API_KEY)
        
        # 현재 모드에 맞는 시스템 선택
        current_mode = st.session_state.get("search_mode", "report")
        selected_retriever = retriever_systems.get(current_mode)
        
        if selected_retriever is None:
            st.error(f"{current_mode} 엔진을 로드할 수 없습니다.")
            st.stop()
            
        with st.spinner("답변 생성 중..."):
            try:
                p_name = "리포트(R5)" if current_mode == "report" else "이폼(E5)"
                
                answer = generate_answer(API_KEY, selected_retriever, prompt_input, current_mode)
                
                # 답변 하단에 출처 표시 추가 (글씨 크기를 줄여 참고용으로 강조)
                final_answer = f"{answer}\n\n---\n> <span style='font-size: 12px; opacity: 0.7;'>📍 현재 답변은 **[{p_name}]** 문서를 바탕으로 작성되었습니다.</span>"
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                error_msg = "현재 API 서버에 문제가 있어 답변을 할 수 없습니다. 관리자에게 문의해 주시기 바랍니다."
                #st.error(error_msg)
                st.error(f"에러 내용: {str(e)}")
                print(f"DEBUG ERROR: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

#st.markdown("<div id='bottom-anchor'></div>", unsafe_allow_html=True)
