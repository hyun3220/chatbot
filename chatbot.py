import streamlit as st
import os
# [수정] 필요한 도구들을 상단에서 명확히 임포트합니다.
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 페이지 설정
st.set_page_config(page_title="클립리포트 AI 챗봇", page_icon="🤖")

# 2. API 키 설정
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    # [핵심] 상단에서 genai 설정을 미리 완료합니다.
    genai.configure(api_key=API_KEY, transport='rest')
except KeyError:
    st.error("서버에 API 키가 설정되지 않았습니다. 관리자에게 문의하세요.")
    st.stop()

# 3. 문서 학습 로직
@st.cache_resource
def get_vectorstore(api_key, pdf_path):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    if not os.path.exists(pdf_path):
        st.error(f"에러: '{pdf_path}' 파일이 존재하지 않습니다.")
        st.stop()
        
    pdf_loader = PyPDFLoader(pdf_path) 
    pdf_docs = pdf_loader.load()
    
    api_url = "https://technet.hancomins.com/board/api/R5/symbols/ReportView.html"
    web_loader = WebBaseLoader(api_url)
    web_docs = web_loader.load()
    
    all_docs = pdf_docs + web_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=None
    )
    return vectorstore

# 4. 답변 생성 로직
def generate_answer(api_key, vectorstore, query):
    # [해결] 함수 내부에서도 안전하게 호출할 수 있도록 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        # v1beta 404 에러를 방지하는 최신 규격 설정
        version="v1",
        transport="rest",
        client_options={"api_endpoint": "https://generativelanguage.googleapis.com"},
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # [이전 프롬프트 유지]
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)' 전문 고객 지원 및 API 개발 가이드 AI 챗봇입니다.\n"
        "당신은 매우 유능한 서포트 엔지니어입니다. 사용자가 묻는 함수명(예: report.setDefaultSavePDFOption)이 Context에 존재하는지 철자 하나하나 대조하며 꼼꼼히 확인하세요.\n"
        "모든 코드는 마크다운 코드 블록 형식을 사용하여 줄바꿈과 들여쓰기를 유지해서 출력하세요.\n"
        "제공된 PDF 문서와 ReportView.html의 API 정보를 바탕으로 사용자의 질문에 답변하세요.\n"
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
        "    \n"
        "    // 👇 여기에 API 적용 예시 작성\n"
        "    \n"
        "    report.view();\n"
        "}}\n"
        "</script>\n"
        "</head>\n"
        "--- [웹 뷰어 스크립트 템플릿 끝] ---\n\n"
        "문서에 없는 내용은 지어내지 말고 '해당 내용을 찾을 수 없습니다'라고 정중히 답하세요.\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 클립리포트 사용법에 대해 무엇이든 물어보세요. 😊"}]

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
