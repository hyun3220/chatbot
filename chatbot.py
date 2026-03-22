import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 페이지 설정
st.set_page_config(page_title="클립리포트 AI 챗봇", page_icon="🤖")

# 2. API 키 설정 (.streamlit/secrets.toml 에서 가져오기)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("서버에 API 키가 설정되지 않았습니다. 관리자에게 문의하세요.")
    st.stop()

# 3. 문서 학습 로직
@st.cache_resource
def get_vectorstore(api_key, pdf_path):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # 벡터 저장소 경로 설정 (영구 저장 가능하도록 설정)
    persist_directory = "./chroma_db"

    # 1단계: 만약 이미 빌드된 DB 폴더(chroma_db)가 있다면? -> 바로 로드!
    if os.path.exists(persist_directory):
        st.info("기존에 생성된 지식 베이스를 불러옵니다.")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # 2단계: DB 폴더가 없다면? -> 넘겨받은 pdf_path(파일명)를 사용해서 새로 구축!
    st.warning("새로운 지식 베이스를 구축합니다. 잠시만 기다려주세요.")
    
    if not os.path.exists(pdf_path):
        st.error(f"에러: '{pdf_path}' 파일이 존재하지 않습니다. 경로를 확인해주세요.")
        st.stop()
        
    # 여기서 실제 pdf_path를 사용하여 로드합니다.
    pdf_loader = PyPDFLoader(pdf_path) 
    pdf_docs = pdf_loader.load()
    
    api_url = "https://technet.hancomins.com/board/api/R5/symbols/ReportView.html"
    web_loader = WebBaseLoader(api_url)
    web_docs = web_loader.load()
    
    all_docs = pdf_docs + web_docs
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 조금 더 촘촘하게 자름
    chunk_overlap=50    # 문맥 연결을 위해 겹치는 구간을 둠
    )

    splits = text_splitter.split_documents(all_docs)
    
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # [추가] 내용이 비어있는(None이거나 공백만 있는) 문서 제거
    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    # 벡터 저장소 생성 및 저장
    vectorstore = Chroma.from_documents(
        documents=valid_documents, # 수정된 변수 사용
        embedding=embeddings,
        persist_directory=None
    )
    return vectorstore

    # return vectorstore

# 4. 답변 생성 로직
def generate_answer(api_key, vectorstore, query):
    # 필요한 라이브러리를 함수 안에서 확실히 임포트
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0,
        safety_settings={
            # 문자열이 아니라 이 '객체' 자체를 키로 써야 합니다.
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)' 전문 고객 지원 및 API 개발 가이드 AI 챗봇입니다."
        "당신은 매우 유능한 서포트 엔지니어입니다. 사용자가 묻는 함수명(예: report.setDefaultSavePDFOption)이 Context에 존재하는지 철자 하나하나 대조하며 꼼꼼히 확인하세요. 비슷한 이름의 함수가 있다면 그것이 사용자가 찾는 것인지 판단하여 답변하세요."
        "모든 코드는 마크다운 코드 블록 형식을 사용하여 줄바꿈과 들여쓰기를 유지해서 출력하세요."
        "제공된 PDF 문서와 ReportView.html의 API 정보를 바탕으로 사용자의 질문에 답변하세요. "
        "사용자가 특정 기능, API, 함수 등에 대해 물어보면 알맞은 ReportView API 함수명과 설명을 친절하게 설명해주세요.\n"
        "만약 뷰어를 호출하지 않고 바로 저장하는 API 함수 알려줘. 라고 질문이 왔을때 인쇄 관련 API를 알려준다거나 하는 잘못된 정보를 제공해서는 안돼. 자세히 잘 찾아서 제공해야해"
        "🚨 [매우 중요 - API 예시 코드 작성 규칙]: "
        "API 사용 예시 코드를 보여줄 때는 단순히 자바스크립트 한 줄만 출력하지 말고, 반드시 아래의 [웹 뷰어 스크립트 템플릿] 구조 안에 해당 API를 어떻게 적용하는지 완전한 형태의 코드로 작성해서 보여주세요.\n\n"
        "--- [웹 뷰어 스크립트 템플릿 시작] ---\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "<title>Report</title>\n"
        "\n"
        "<script type='text/javascript'>\n"
        "var report; // 외부에서 API를 호출하기 위해 전역 변수로 선언할 수 있습니다.\n\n"
        "function html2xml(divPath){{\n"
        "    var reportkey = \"<%=resultKey%>\";\n"
        "    report = createReport(\"./report_server.jsp\", reportkey, document.getElementById(divPath));\n"
        "    report.setStyle(\"close_button\", \"display:none;\");\n"
        "    \n"
        "    // 👇 여기에 API 적용 예시 작성\n"
        "    \n"
        "    report.view();\n"
        "}}\n"
        "// 👇 버튼 클릭 함수 등 커스텀 JS 작성\n"
        "</script>\n"
        "</head>\n"
        "<body onload=\"html2xml('targetDiv1')\">\n"
        "\n"
        "<div id='targetDiv1' style='position:absolute;top:5px;left:5px;right:5px;bottom:5px;'>\n"
        "    <span style=\"visibility: hidden; font-family:나눔고딕\">.</span>\n"
        "    <span style=\"visibility: hidden; font-family:NanumGothic\">.</span>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
        "--- [웹 뷰어 스크립트 템플릿 끝] ---\n\n"
        "문서에 없는 내용은 지어내지 말고 '해당 내용을 찾을 수 없습니다'라고 정중히 답하세요. "
        "가독성 좋게 작성하세요.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# ==========================================
# 5. 웹 UI 화면 그리기 (아까 통째로 날아가 버렸던 녀석들입니다 🥲)
# ==========================================
st.title("🤖 클립리포트 v5.0 & API 전문 챗봇")

with st.sidebar:
    st.header("ℹ️ 챗봇 정보")
    st.write("이 챗봇은 아래의 공식 문서를 바탕으로 답변을 제공합니다.")
    st.markdown("---")
    st.markdown("📄 **학습된 지식 베이스**")
    st.markdown("- `CLIP Report 5.0 매뉴얼`")
    st.markdown("- `ReportView API`")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 클립리포트 매뉴얼 내용이나 ReportView API 사용법에 대해 무엇이든 물어보세요. 😊"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pdf_file_path = "클립리포트 v5.0 매뉴얼.pdf" 
        
        if not os.path.exists(pdf_file_path):
            st.error("시스템 오류: 서버에서 PDF 문서를 찾을 수 없습니다.")
            st.stop()
            
        vectorstore = get_vectorstore(API_KEY, pdf_file_path)
        
        with st.spinner("지식 창고(매뉴얼+API)를 검색하여 답변을 생성하고 있습니다..."):
            answer = generate_answer(API_KEY, vectorstore, prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
