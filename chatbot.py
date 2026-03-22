import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# [2026년 최신 구조] langchain 패키지를 거치지 않고 직접 라이브러리에서 호출
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# 1. 페이지 설정
st.set_page_config(page_title="클립리포트 AI 챗봇", page_icon="🤖")

# 2. API 키 설정
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("서버에 API 키가 설정되지 않았습니다. 관리자에게 문의하세요.")
    st.stop()

# 3. 문서 학습 로직 (캐싱 처리)
@st.cache_resource
def get_vectorstore(api_key, pdf_path):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # [수정] 웹 크롤링 시 차단 방지를 위한 User-Agent 설정
    import langchain_community.document_loaders.web_base
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    # 1단계: PDF 로드
    if not os.path.exists(pdf_path):
        st.error(f"에러: '{pdf_path}' 파일이 존재하지 않습니다.")
        st.stop()
    
    pdf_loader = PyPDFLoader(pdf_path) 
    pdf_docs = pdf_loader.load()
    
    # 2단계: 웹 API 문서 로드
    api_url = "https://technet.hancomins.com/board/api/R5/symbols/ReportView.html"
    web_loader = WebBaseLoader(api_url)
    web_docs = web_loader.load()
    
    all_docs = pdf_docs + web_docs
    
    # 3단계: 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs) # 여기서 'splits'라는 이름으로 저장됨
    
    # 4단계: [중요 수정] 'splits' 변수를 사용하여 빈 내용 필터링
    # 아까 에러가 났던 부분: 'documents'가 아니라 'splits'를 참조해야 합니다.
    valid_documents = [doc for doc in splits if doc.page_content and doc.page_content.strip()]
    
    # 5단계: 벡터 저장소 생성 (메모리 모드로 설정하여 DB 버전 충돌 방지)
    # persist_directory를 사용하면 서버 환경에 따라 에러가 날 수 있으므로 None 권장
    vectorstore = Chroma.from_documents(
        documents=valid_documents, 
        embedding=embeddings,
        persist_directory=None 
    )
    return vectorstore

# 4. 답변 생성 로직
def generate_answer(api_key, vectorstore, query):
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # k값을 조금 늘려 정확도 향상
    
    system_prompt = (
        "당신은 '클립리포트(CLIP report v5.0)' 전문 고객 지원 및 API 개발 가이드 AI 챗봇입니다.\n"
        "당신은 매우 유능한 서포트 엔지니어입니다. 사용자가 묻는 함수명이 Context에 존재하는지 꼼꼼히 확인하세요.\n"
        "제공된 지식 베이스만을 바탕으로 답변하고, 없는 내용은 지어내지 마세요.\n\n"
        "{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# 5. 웹 UI 화면 그리기
st.title("🤖 클립리포트 v5.0 & API 전문 챗봇")

with st.sidebar:
    st.header("ℹ️ 챗봇 정보")
    st.write("공식 매뉴얼과 API 문서를 기반으로 답변합니다.")
    if st.button("🔄 지식 베이스 새로고침"):
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 클립리포트 v5.0에 대해 궁금한 점을 물어보세요. 😊"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        pdf_file_path = "클립리포트 v5.0 매뉴얼.pdf" 
        
        # 1. 벡터스토어 로드
        vectorstore = get_vectorstore(API_KEY, pdf_file_path)
        
        # 2. 답변 생성
        with st.spinner("답변 생성 중..."):
            try:
                answer = generate_answer(API_KEY, vectorstore, user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
