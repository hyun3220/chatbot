# 상단 임포트 부분 수정 (기존꺼 다 지우고 이걸로 바꿔주세요)
import streamlit as st
import os

# 환경 변수로 v1을 강제 고정
os.environ["GOOGLE_API_VERSION"] = "v1"

import google.generativeai as genai
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
    </style>

""", unsafe_allow_html=True)

# ==========================================
# 사이드바 구성 추가
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📎 CLIP Report 5.0 AI</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 0px; opacity: 0.2;'>", unsafe_allow_html=True)
    st.header("⚙️ 설정 및 정보")
    st.write("CLIP Report 5.0 매뉴얼 기반 지능형 챗봇입니다. API 가이드 및 예제 코드를 제공합니다.")
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
def get_retriever(API_KEY, pdf_path):
    CHROMA_PERSIST_DIR = "./chroma_db"
    
    # 1. 문서 로드 및 분할 (기존과 동일)
    urls = [
        "https://clipdocs.com/api"
    ]
    docs = []
    
    # 웹 페이지 로드
    for url in urls:
        st.sidebar.text(f"URL 로딩 중: {url}")
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    
    # PDF 로드 (파일이 있는 경우에만 처리)
    if os.path.exists('clipreport.pdf'):
        st.sidebar.text("PDF 로딩 중: clipreport.pdf")
        pdf_loader = PyPDFLoader('clipreport.pdf')
        docs.extend(pdf_loader.load())
        
    # 문서를 크게 쪼개기 (문맥 유지)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs = text_splitter.split_documents(docs)
    
    # 2. 리트리버 세팅
    try:
        # 키워드 기반 BM25 리트리버
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 10  
        
        # 의미 기반 Chroma 벡터 리트리버
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            ),
            persist_directory=CHROMA_PERSIST_DIR
        )
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 🚀 랭체인 패키지 에러 우회! 직접 하이브리드 엔진 함수 생성
        def custom_hybrid_search(query: str):
            # 두 검색 엔진에서 각각 결과를 가져옴
            docs_bm25 = bm25_retriever.invoke(query)
            docs_chroma = chroma_retriever.invoke(query)
            
            # 중복을 없애고 두 결과를 지퍼(zip)처럼 교차 결합!
            merged_docs = []
            seen_content = set()
            
            for d1, d2 in zip(docs_bm25 + [None]*10, docs_chroma + [None]*10):
                if d1 and d1.page_content not in seen_content:
                    seen_content.add(d1.page_content)
                    merged_docs.append(d1)
                if d2 and d2.page_content not in seen_content:
                    seen_content.add(d2.page_content)
                    merged_docs.append(d2)
                    
            return merged_docs[:15] # 상위 15개 텍스트 리턴

        # 만든 무적 함수를 Runnable로 던져줍니다!
        return RunnableLambda(custom_hybrid_search)
        
    except Exception as e:
        st.error(f"벡터 DB 초기화 오류: {e}")
        return None

# 답변 생성 로직
def generate_answer(api_key, retriever, query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=api_key, temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    system_prompt = (
        "당신은 고객님의 모호한 질문을 공식 API 명칭으로 변환하여 답변하는 **천재적인 매핑 엔지니어**입니다.\n\n"
        "🔍 [문맥 추론 규칙]:\n"
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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        pdf_path = "클립리포트 v5.0 매뉴얼.pdf"
        retriever_engine = get_retriever(API_KEY, pdf_path)
        
        with st.spinner("답변 생성 중..."):
            try:
                answer = generate_answer(API_KEY, retriever_engine, prompt_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = "현재 API 서버에 문제가 있어 답변을 할 수 없습니다. 관리자에게 문의해 주시기 바랍니다."
                #st.error(error_msg)
                st.error(f"에러 내용: {str(e)}")
                print(f"DEBUG ERROR: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

#st.markdown("<div id='bottom-anchor'></div>", unsafe_allow_html=True)
