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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)
    
    return Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. 답변 생성 로직
def generate_answer(api_key, vectorstore, query):
    # 로그에서 'Unexpected argument'라고 했던 version, transport를 제거하고 
    # 환경 변수(v1)의 힘을 믿고 깔끔하게 선언합니다.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
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
    
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)'의 모든 기술 문서를 완벽히 숙지한 **수석 기술 지원 엔지니어**입니다.\n"
        "사용자가 아주 짧거나 모호하게 질문하더라도, 당신은 질문의 의도를 정확히 파악하여 전문적인 답변을 제공해야 합니다.\n\n"
        
        "💡 [답변 원칙]:\n"
        "1. **의도 파악**: 질문이 짧다면(예: 'PDF 인쇄'), 사용자가 '웹 브라우저의 인쇄 기능을 쓰려는 것인지' 혹은 '서버에서 직접 PDF를 생성하려는 것인지' 맥락을 구분하여 가장 적절한 API를 추천하세요.\n"
        "2. **정확성**: 함수명(예: report.print)은 대소문자와 철자를 Context와 대조하여 100% 일치할 때만 답변하세요. 확실하지 않다면 '비슷한 기능을 가진 함수'임을 명시하세요.\n"
        "3. **친절한 가이드**: 단순히 함수만 띡 던지지 말고, '이 함수는 어떤 상황에서 주로 사용되는지' 한 줄 설명을 덧붙이세요.\n\n"
        
        "🚨 [매우 중요 - 답변 구성 순서]:\n"
        "사용자가 무엇을 물어보든 항상 아래 순서로 답변하세요.\n"
        "1️⃣ **기능 요약**: 질문한 기능에 대한 짧은 정의\n"
        "2️⃣ **핵심 API**: 정확한 함수명과 파라미터 설명\n"
        "3️⃣ **전체 예시 코드**: 제공된 [웹 뷰어 스크립트 템플릿]을 활용한 완성된 HTML 코드\n"
        "4️⃣ **주의 사항**: 해당 API 사용 시 개발자가 자주 실수하는 부분이나 팁\n\n"
        
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
        "만약 사용자가 PDF 저장 옵션 함수 알려줘 라고 질문할 경우 저장 선택 옵션이 아니라 말 그대로 PDF를 저장할 때 PDF의 옵션을 지정하는 함수를 질문한 거니까 의도를 정확히 파악하세요."
        "문서에 없는 내용은 절대 지어내지 마세요. 모르는 내용은 '매뉴얼에서 해당 내용을 찾을 수 없으니 공식 기술 지원팀에 문의해 주세요'라고 정중히 안내하세요.\n\n"
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
