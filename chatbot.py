import streamlit as st
import os

# [최종 해결책] 라이브러리가 딴소리 못하게 환경 변수로 v1을 강제 고정합니다.
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

# 1. 페이지 설정
st.set_page_config(page_title="클립리포트 AI 챗봇", page_icon="🤖")

# 2. API 키 설정
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
except KeyError:
    st.error("서버에 API 키가 설정되지 않았습니다.")
    st.stop()

# 3. 문서 학습 로직 (캐싱 적용)
@st.cache_resource
def get_vectorstore(api_key, pdf_path):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    if not os.path.exists(pdf_path):
        st.error(f"'{pdf_path}' 파일이 없습니다.")
        st.stop()
        
    pdf_loader = PyPDFLoader(pdf_path)
    all_docs = pdf_loader.load()
    
    # 웹 데이터 추가 (선택 사항)
    try:
        web_loader = WebBaseLoader("https://technet.hancomins.com/board/api/R5/symbols/ReportView.html")
        all_docs += web_loader.load()
    except:
        pass

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. 답변 생성 로직
def generate_answer(api_key, vectorstore, query):
    # 로그에서 'Unexpected argument'라고 했던 version, transport를 제거하고 
    # 환경 변수(v1)의 힘을 믿고 깔끔하게 선언합니다.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001", 
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
    
    # [요청하신 상세 프롬프트 그대로 유지]
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)' 전문 고객 지원 및 API 개발 가이드 AI 챗봇입니다.\n"
        "당신은 매우 유능한 서포트 엔지니어입니다. 사용자가 묻는 함수명(예: report.setDefaultSavePDFOption)이 Context에 존재하는지 철자 하나하나 대조하며 꼼꼼히 확인하세요. 비슷한 이름의 함수가 있다면 그것이 사용자가 찾는 것인지 판단하여 답변하세요.\n"
        "모든 코드는 마크다운 코드 블록 형식을 사용하여 줄바꿈과 들여쓰기를 유지해서 출력하세요.\n"
        "제공된 PDF 문서와 ReportView.html의 API 정보를 바탕으로 사용자의 질문에 답변하세요.\n"
        "만약 뷰어를 호출하지 않고 바로 저장하는 API 함수 알려줘. 라고 질문이 왔을때 인쇄 관련 API를 알려준다거나 하는 잘못된 정보를 제공해서는 안돼.\n"
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
        "--- [웹 뷰어 스크립트 템플릿 끝] ---\n\n"
        "문서에 없는 내용은 지어내지 말고 '해당 내용을 찾을 수 없습니다'라고 정중히 답하세요.\n\n"
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

# 5. UI 및 실행 로직
st.title("🤖 클립리포트 v5.0 전문 챗봇")

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
        vectorstore = get_vectorstore(API_KEY, pdf_path)
        with st.spinner("답변 생성 중..."):
            try:
                answer = generate_answer(API_KEY, vectorstore, prompt_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
