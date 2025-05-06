# main.py
import streamlit as st
import requests
import tiktoken
import os

from loguru import logger
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
import docx
from io import BytesIO


# --------------------------
# 기능 함수
# --------------------------
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
            
        if '.pdf' in file_name:
            loader = PyPDFLoader(file_name)
        elif '.docx' in file_name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in file_name:
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue
            
        documents = loader.load_and_split()
        doc_list.extend(documents)
        
    return doc_list

# 텍스트 청킹 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# 벡터 스토어 생성 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 대화 체인 생성 함수
def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

# 채팅 기록 저장 함수
def save_chat_history(title=""):
    if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        if not os.path.exists('chat_history'):
            os.makedirs('chat_history')
        
        if not title:
            title = "chat_history"
        
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        filename = f"{safe_title}.txt"
        filepath = os.path.join('chat_history', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for msg in st.session_state.messages:
                f.write(f"[{msg['role'].upper()}] {msg['content']}\n")
        
        st.success(f"✅ 채팅 기록이 저장되었습니다: {filename}")
        return True
    else:
        st.warning("저장할 채팅 기록이 없습니다.")
        return False
    
# 저장된 채팅 기록 표시 함수
def display_saved_chats():
    st.subheader("📁 저장된 채팅 기록")
    
    if not os.path.exists('chat_history'):
        st.info("채팅 기록 폴더가 존재하지 않습니다.")
        return []
        
    files = [f for f in os.listdir('chat_history') if f.endswith('.txt')]
    
    if not files:
        st.info("아직 저장된 채팅 기록이 없습니다.")
        return []
        
    cols = st.columns(3)
    for idx, file in enumerate(files):
        with cols[idx%3]:
            with open(os.path.join('chat_history', file), 'r', encoding='utf-8') as f:
                content = f.read()
            st.download_button(
                label=f"📄 {file}",
                data=content,
                file_name=file,
                mime="text/plain"
            )
    
    return files

# 채팅 기록 불러오기 함수
def load_chat_history(filename):
    messages = []
    file_path = os.path.join('chat_history', filename)
    
    if not os.path.exists(file_path):
        st.error(f"{filename} 파일을 찾을 수 없습니다.")
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("[USER]"):
                messages.append({"role": "user", "content": line[len("[USER] "):].strip()})
            elif line.startswith("[ASSISTANT]"):
                messages.append({"role": "assistant", "content": line[len("[ASSISTANT] "):].strip()})
                
    if messages:
        st.session_state['messages'] = messages
        st.success(f"✅ {filename} 채팅 기록을 불러왔습니다.")
        return True
    
    st.warning(f"{filename} 파일에서 메시지를 찾을 수 없습니다.")
    return False    

def process_file(uploaded_file):
    """PDF/DOCX/TXT 파일에서 텍스트 추출"""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return ""

def google_search(query, api_key, cse_id, num=5):
    """Google Custom Search API 호출"""
    service = build("customsearch", "v1", developerKey=api_key)
    return service.cse().list(q=query, cx=cse_id, num=num).execute().get("items", [])

def generate_answer(messages, openai_key, model="gpt-4"):
    """OpenAI Chat Completion 호출"""
    headers = {"Authorization": f"Bearer {openai_key}"}
    payload = {"model": model, "messages": messages}
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=payload
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]



def setup_sidebar():
    """사이드바에서 모드·API 키·파일 업로드 설정"""
    st.sidebar.header("⚙️ Config")
    st.sidebar.image(
        "./36logo.png",
        use_container_width=True
    )
    
    with st.sidebar.expander("🤪 Contributors", expanded=False):
        st.markdown("""
                    ## Leader: 😝박기호호호
                    ### Member1 : 🤑김태윤
                    ### Memeber2 : 👷‍♂️박현신 """)
        
    
    st.sidebar.markdown("""\nI've been workin like a dog""")
    st.sidebar.divider()
    mode = st.sidebar.radio(
        "Mode Select",
        ["Web Searching Mode", "Uploaded File + Searching Mode", "Answer Based on Uploaded File Mode"]
    )
    
    st.sidebar.divider()
    
    # 채팅 기록 관리 섹션 - 사이드바로 모두 통합
    st.sidebar.subheader("💬 Chat History Manager")
    
    # 채팅 저장 기능
    with st.sidebar.expander("Save Current Chat", expanded=False):
        chat_title = st.text_input("Chat Title", "", key="save_chat_title")
        if st.button("Save Chat History"):
            save_chat_history(chat_title)
    
    # 채팅 불러오기 기능
    with st.sidebar.expander("Load Chat History", expanded=False):
        if not os.path.exists('chat_history'):
            st.info("채팅 기록 폴더가 존재하지 않습니다.")
        else:
            files = [f for f in os.listdir('chat_history') if f.endswith('.txt')]
            if not files:
                st.info("아직 저장된 채팅 기록이 없습니다.")
            else:
                selected_file = st.selectbox("Select saved chat", [""] + files)
                if st.button("Load Selected Chat") and selected_file:
                    load_chat_history(selected_file)
    
    # 저장된 채팅 기록 표시 기능
    with st.sidebar.expander("Browse Saved Chats", expanded=False):
        if os.path.exists('chat_history'):
            files = [f for f in os.listdir('chat_history') if f.endswith('.txt')]
            if files:
                for file in files:
                    with open(os.path.join('chat_history', file), 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.download_button(
                        label=f"📄 {file}",
                        data=content,
                        file_name=file,
                        mime="text/plain"
                    )
            else:
                st.info("아직 저장된 채팅 기록이 없습니다.")
        else:
            st.info("채팅 기록 폴더가 존재하지 않습니다.")
    
    st.sidebar.divider()
    
    # API 키 및 파일 업로드 부분은 그대로 유지
    openai_key = st.sidebar.text_input("OpenAI API 키", type="password")
    google_key = google_cse = None
    if "Searching" in mode:
        google_key = st.sidebar.text_input("Google API 키", type="password")
        google_cse = st.sidebar.text_input("Google CSE ID", type="password")
    uploader = st.sidebar.file_uploader(
        "파일 업로드 (pdf/docx/txt)", type=["pdf", "docx", "txt"]
    )
    if uploader:
        st.session_state.file_content = process_file(uploader)
        st.sidebar.success("파일 분석 완료!")

    return mode, openai_key, google_key, google_cse


# --------------------------
# 메인 실행 함수
# --------------------------
def main():
    st.set_page_config(
        page_title="Help Me Please....",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="auto"
    )
    # 세션 초기화
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_content" not in st.session_state:
        st.session_state.file_content = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    placeholder = st.empty()

    # 초기 화면
    if not st.session_state.chat_started:
        with placeholder.container():
            st.image("36logo.png", width=50)
            st.markdown("## 무엇이 궁금하신가요?")
            setup_sidebar()
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                q = st.text_input(
                    "", placeholder="질문을 입력하고 Enter를 누르세요", key="init_q"
                )
                if q:
                    st.session_state.messages.append({
                        "role": "user", "content": q
                    })
                    st.session_state.chat_started = True
                    # 초기 화면 제거 후 재실행
                    placeholder.empty()
                    st.rerun()
    else:
        # 초기 화면 placeholder 제거
        placeholder.empty()

        # 사이드바 설정
        mode, openai_key, google_key, google_cse = setup_sidebar()
        st.title("🔍 AI 검색 엔진")

        # 첫 인사
        if (len(st.session_state.messages) == 1 
            and st.session_state.messages[0]["role"] == "user"):
            st.session_state.messages.insert(0, {
                "role": "assistant",
                "content": "안녕하세요! 무엇이든 물어보세요."
            })

        # 채팅 기록 표시
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # 추가 질문 입력
        prompt = st.chat_input("추가 질문을 입력하세요...")
        if prompt:
            # API 키 검증
            if not openai_key or ("검색" in mode and (not google_key or not google_cse)):
                st.sidebar.error("필수 API 키를 모두 입력해주세요")
            else:
                # 사용자 메시지 기록
                st.session_state.messages.append({
                    "role": "user", "content": prompt
                })

                # 컨텍스트 구성
                context = ""
                if "File" in mode and st.session_state.file_content:
                    context += f"파일 내용:\n{st.session_state.file_content}\n\n"
                sources = []
                if "Searching" in mode:
                    results = google_search(prompt, google_key, google_cse, num=3)
                    for i, item in enumerate(results, 1):
                        context += f"[{i}] {item['title']}\n{item.get('snippet','')}\n\n"
                        sources.append((item['title'], item['link']))

                # OpenAI 호출
                messages = [
                    {"role": "system", "content": "최대한 친근하게 대답하세요."},
                    {"role": "user", "content": f"{prompt}\n\n{context}"}
                ]
                answer = generate_answer(messages, openai_key)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer
                })

                # 화면 갱신
                st.rerun()

if __name__ == "__main__":
    main()
