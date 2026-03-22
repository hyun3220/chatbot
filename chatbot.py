import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
# [핵심] 에러 나는 chains 대신 사용하는 최신 모듈
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 페이지 설정
st.set_page_config(page_title="클립리포트 AI 챗봇", page_icon="🤖")

# 2. API 키 설정
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
    
    # 웹 크롤링 차단 방지
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    if not os.path.exists(pdf_path):
        st.error(f"에러: '{pdf_path}' 파일이 존재하지 않습니다.")
        st.stop()
        
    # PDF 및 웹 데이터 로드
    pdf_loader = PyPDFLoader(pdf_path) 
    pdf_docs = pdf_loader.load()
    
    api_url = "https://technet.hancomins.com/board/api/R5/symbols/ReportView.html"
    web_loader = WebBaseLoader(api_url)
    web_docs = web_loader.load()
    
    all_docs = pdf_docs + web_docs
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    # 빈 문서 제거
    valid_documents = [doc for doc in splits if doc.page_content and doc.page_content.strip()]
    
    # 벡터 저장소 생성 (메모리 모드)
    vectorstore = Chroma.from_documents(
        documents=valid_documents, 
        embedding=embeddings,
        persist_directory=None
    )
    return vectorstore

# 4. 답변 생성 로직 (LCEL 방식 적용)
def generate_answer(api_key, vectorstore, query):
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=api_key,
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # [복구] 예전에 사용하시던 상세 프롬프트 그대로 적용
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)' 전문 고객 지원 및 API 개발 가이드 AI 챗봇입니다.\n"
        "당신은 매우 유능한 서포트 엔지니어입니다. 사용자가 묻는 함수명(예: report.setDefaultSavePDFOption)이 Context에 존재하는지 철자 하나하나 대조하며 꼼꼼히 확인하세요. 비슷한 이름의 함수가 있다면 그것이 사용자가 찾는 것인지 판단하여 답변하세요.\n"
        "모든 코드는 마크다운 코드 블록 형식을 사용하여 줄바꿈과 들여쓰기를 유지해서 출력하세요.\n"
        "제공된 PDF 문서와 ReportView.html의 API 정보를 바탕으로 사용자의 질문에 답변하세요.\n"
        "사용자가 특정 기능, API, 함수 등에 대해 물어보면 알맞은 ReportView API 함수명과 설명을 친절하게 설명해주세요.\n"
        "만약 뷰어를 호출하지 않고 바로 저장하는 API 함수 알려줘. 라고 질문이 왔을때 인쇄 관련 API를 알려준다거나 하는 잘못된 정보를 제공해서는 안돼. 자세히 잘 찾아서 제공해야해\n"
        "🚨 [매우 중요 - API 예시 코드 작성 규칙]: \n"
        "API 사용 예시 코드를 보여줄 때는 반드시 아래의 [웹 뷰어 스크립트 템플릿] 구조 안에 해당 API를 어떻게 적용하는지 완전한 형태의 코드로 작성해서 보여주세요.\n\n"
        "--- [웹 뷰어 스크립트 템플릿 시작] ---\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "<title>Report</title>\n"
        "<script type='text/javascript'>\n"
        "var report;\n"
        "function html2xml(divPath){{\n"
        "    var reportkey = \"<%=resultKey%>\";\n"
        "    report = createReport(\"./report_server.jsp\", reportkey, document.getElementById(divPath));\n"
        "    report.setStyle(\"close_button\", \"display:none;\");\n"
        "    \n"
        "    // 👇 여기에 API 적용 예시 작성\n"
        "    \n"
        "    report.view();\n"
        "}}\n"
        "</script>\n"
        "</head>\n"
        "<body onload=\"html2xml('targetDiv1')\">\n"
        "<div id='targetDiv1' style='position:absolute;top:5px;left:5px;right:5px;bottom:5px;'>\n"
        "    <span style=\"visibility: hidden; font-family:나눔고딕\">.</span>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
        "--- [웹 뷰어 스크립트 템플릿 끝] ---\n\n"
        "문서에 없는 내용은 지어내지 말고 '해당 내용을 찾을 수 없습니다'라고 정중히 답하세요.\n"
        "가독성 좋게 작성하세요.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # [수정] 파이썬 3.14에서 에러 없는 LCEL 방식 체인 연결
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(query)

# 5. 웹 UI 화면
st.title("🤖 클립리포트 v5.0 & API 전문 챗봇")

with st.sidebar:
    st.header("ℹ️ 챗봇 정보")
    st.write("공식 매뉴얼과 API 문서를 기반으로 답변합니다.")
    if st.button("🔄 지식 베이스 새로고침"):
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 클립리포트 매뉴얼 내용이나 ReportView API 사용법에 대해 무엇이든 물어보세요. 😊"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        pdf_file_path = "클립리포트 v5.0 매뉴얼.pdf" 
        
        vectorstore = get_vectorstore(API_KEY, pdf_file_path)
        
        with st.spinner("답변 생성 중..."):
            try:
                answer = generate_answer(API_KEY, vectorstore, prompt_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
