import streamlit as st
import os

# [환경 설정] 라이브러리 충돌 방지를 위해 v1 API 강제 고정
os.environ["GOOGLE_API_VERSION"] = "v1"

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 페이지 설정 및 세련된 블랙 & 오렌지 커스텀 CSS
st.set_page_config(page_title="CLIP Report 5.0 AI 어시스턴트", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    /* 전체 배경 및 텍스트 색상 */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* 헤더 및 타이틀 스타일 */
    h1 {
        color: #FF4B2B !important; /* 주황색 포인트 */
        font-weight: 800;
        border-bottom: 2px solid #FF4B2B;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }

    /* 채팅 메시지 박스 */
    .stChatMessage {
        background-color: #1A1C23 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin-bottom: 10px !important;
        border: 1px solid #333;
    }
    
    /* 사용자 메시지 바 왼쪽 포인트 */
    [data-testid="stChatMessageUser"] {
        border-left: 5px solid #FF4B2B !important;
    }

    /* 어시스턴트 메시지 바 왼쪽 포인트 */
    [data-testid="stChatMessageAssistant"] {
        border-left: 5px solid #555 !important;
    }

    /* 입력창 디자인 */
    div[data-testid="stChatInput"] {
        border: 2px solid #FF4B2B !important;
        border-radius: 30px !important;
        background-color: #000000 !important;
    }
    
    /* 로딩 스피너 색상 */
    .stSpinner > div {
        border-top-color: #FF4B2B !important;
    }
    
    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
        background-color: #111418 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. API 키 설정
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    st.error("🔑 서버에 API 키가 설정되지 않았습니다 (Secrets 확인 필요).")
    st.stop()

# 3. 문서 학습 로직 (검색 정확도를 위해 Chunk Size 및 Overlap 최적화)
@st.cache_resource
def get_vectorstore(api_key, pdf_path):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' 파일이 없습니다.")
        st.stop()
        
    pdf_loader = PyPDFLoader(pdf_path)
    all_docs = pdf_loader.load()
    
    # 웹 데이터 추가
    try:
        web_loader = WebBaseLoader("https://technet.hancomins.com/board/api/R5/symbols/ReportView.html")
        all_docs += web_loader.load()
    except:
        pass

    # 문맥 유지를 위해 chunk_size를 늘리고, 끊김 방지를 위해 overlap을 조정했습니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. 답변 생성 로직 (지능형 문맥 추론 프롬프트 적용)
def generate_answer(api_key, vectorstore, query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001", # 안정적인 실행을 위한 모델 버전 고정
        google_api_key=api_key,
        temperature=0,
        safety_settings={cat: HarmBlockThreshold.BLOCK_NONE for cat in HarmCategory}
    )
    
    # 검색 결과(k)를 8개로 늘려 더 폭넓은 문맥을 AI에게 전달합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)'의 수석 기술 지원 엔지니어입니다.\n"
        "사용자의 모호한 일상 용어를 정확한 API 명칭으로 매핑하여 답변하는 능력이 매우 뛰어납니다.\n\n"
        
        "🔍 [문맥 추론 및 매핑 규칙]:\n"
        "- 사용자가 '순서 바꾸기', '위치 변경', '정렬' 등을 말하면: 'Order', 'Index', 'Sequence', 'Sort', 'Menu' 관련 키워드를 검색하세요.\n"
        "- 예: '저장 순서 바꾸기' -> 문서의 '디폴트 순서를 지정하는 옵션(setReportSaveMenuOrder)'을 찾아야 함.\n"
        "- 사용자가 '저장 옵션', '내보내기' 등을 말하면: 'Save', 'Export', 'Format' 관련 함수를 우선 탐색하세요.\n"
        "- 사용자가 '숨기기', '안 보이게' 등을 말하면: 'Visible', 'Display', 'Hide' 관련 기능을 찾으세요.\n"
        "- 질문의 단어와 문서의 단어가 100% 일치하지 않아도 **기능적 의도**가 같다면 그 함수를 정답으로 제시하세요.\n\n"
        
        "💡 [답변 원칙]:\n"
        "1. **의도 파악**: 질문이 짧다면 맥락을 유추하여 가장 적절한 API를 추천하세요.\n"
        "2. **정확성**: 함수명은 대소문자와 철자를 Context와 100% 대조하여 답변하세요.\n"
        "3. **친절한 가이드**: 사용 상황에 대한 한 줄 설명을 반드시 포함하세요.\n\n"
        
        "🚨 [답변 구성 순서]:\n"
        "1️⃣ **기능 요약**: 질문한 기능에 대한 짧은 정의\n"
        "2️⃣ **핵심 API**: 정확한 함수명과 파라미터 설명\n"
        "3️⃣ **전체 예시 코드**: 제공된 [웹 뷰어 스크립트 템플릿]을 활용한 코드\n"
        "4️⃣ **주의 사항**: 개발자 팁 또는 자주 발생하는 실수\n\n"
        
        "--- [웹 뷰어 스크립트 템플릿] ---\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "<script type='text/javascript'>\n"
        "var report;\n"
        "function html2xml(divPath){{\n"
        "    var reportkey = \"<%=resultKey%>\";\n"
        "    report = createReport(\"./report_server.jsp\", reportkey, document.getElementById(divPath));\n"
        "    \n"
        "    // 👇 여기에 사용자가 질문한 API 적용\n"
        "    \n"
        "    report.view();\n"
        "}}\n"
        "</script>\n"
        "</head>\n"
        "<body><div id='targetDiv1'></div></body>\n"
        "</html>\n"
        "--- [템플릿 끝] ---\n\n"
        
        "문서에 없는 내용은 절대 지어내지 마세요. 모르는 내용은 기술 지원팀 문의를 안내하세요.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(query)

# 5. UI 및 대화 실행 로직
st.title("🤖 CLIP Report 전문 어시스턴트")

# 사이드바 구성
with st.sidebar:
    st.header("설정 및 정보")
    st.write("클립리포트 5.0 매뉴얼 기반 지능형 챗봇입니다.")
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

# 기존 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        pdf_path = "클립리포트 v5.0 매뉴얼.pdf"
        vectorstore = get_vectorstore(API_KEY, pdf_path)
        with st.spinner("최적의 솔루션을 찾는 중입니다..."):
            try:
                answer = generate_answer(API_KEY, vectorstore, prompt_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
