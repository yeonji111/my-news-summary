import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 스트림릿 페이지 기본 설정
st.set_page_config(page_title="경제 뉴스 요약 봇", layout="wide")

# 🔥 [필수 설정 1] 방금 발급받은 네이버 API 키를 입력하세요.
NAVER_CLIENT_ID = "발급받은 네이버 클라이언트 키 입력"
NAVER_CLIENT_SECRET = "발급 받은 네이커 클라이언트 시크릿 입력"

# 🔥 [필수 설정 2] 발급받은 Groq API 키를 입력하세요.
GROQ_API_KEY = "발급받은 Groq API 키를 입력"

st.title("📰 네이버 정식 API 기반 실시간 경제 뉴스 요약 봇")
st.subheader("3년 차 SM 개발자의 일주일 완성 사이드 프로젝트 (최종 완성본)")

if "news_data" not in st.session_state:
    st.session_state.news_data = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ----------------------------------------------------
# 1단계: 차단 0% 정식 네이버 검색 API 기반 뉴스 수집
# ----------------------------------------------------
def get_naver_api_news():
    # '경제' 키워드로 가장 최근 뉴스를 10개 정식 요청합니다.
    url = "https://openapi.naver.com/v1/search/news.json?query=경제&display=30&sort=date"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    news_list = []
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("items", []):
                # HTML 태그 제거 정제
                title = BeautifulSoup(item["title"], "html.parser").text.strip()
                description = BeautifulSoup(item["description"], "html.parser").text.strip()
                link = item["link"]
                
                news_list.append({
                    "office": "네이버 제휴 언론사",
                    "title": title,
                    "link": link,
                    "content": description if description else "상세 내용 없음"
                })
        else:
            st.error(f"네이버 API 연동 실패 (상태 코드: {response.status_code}). 키 값을 확인해 주세요.")
    except Exception as e:
        st.error(f"뉴스 수집 중 시스템 에러 발생: {e}")
        
    return news_list

# ----------------------------------------------------
# 2단계: LangChain을 활용한 로컬 Vector DB 구축 및 요약
# ----------------------------------------------------
def init_langchain_rag(news_list):
    documents = []
    for news in news_list:
        doc = Document(
            page_content=f"언론사: {news['office']}\n제목: {news['title']}\n본문: {news['content']}",
            metadata={"source": news['link'], "title": news['title']}
        )
        documents.append(doc)
        
    with st.spinner("한국어 AI 문맥 분석 모델 초기화 중..."):
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vector_db = FAISS.from_documents(documents, embeddings)
        
    all_texts = "\n\n".join([f"[{n['office']}] {n['title']}: {n['content']}..." for n in news_list])
    
    llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 유능한 경제 전문가입니다. 제공된 뉴스 데이터를 바탕으로 오늘 가장 중요한 경제 이슈 3가지를 선정하여 브리핑을 작성해주세요.
        
        [작성 규칙]:
        1. 각 이슈는 이모지와 함께 **굵은 제목**으로 시작하세요.
        2. 각 이슈 아래에는 불릿 포인트(-)를 사용하여 핵심 내용을 2~3줄로 설명하세요.
        3. 전문 용어는 쉽게 풀어서 설명하고, 마크다운(Markdown) 형식을 사용하여 가독성 있게 출력하세요.
        4. 반드시 한국어로 답변하세요.
        
        출력 예시:
        ### 🚀 1. 제목입니다
        - 첫 번째 요약 내용입니다.
        - 두 번째 요약 내용입니다.
        """)
    ])
    
    chain = summary_prompt | llm
    summary_result = chain.invoke({"input": all_texts})
    
    return vector_db, summary_result.content

# ----------------------------------------------------
# 화면 레이아웃
# ----------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📌 수집된 경제 뉴스 & 실시간 브리핑")
    if st.button("🔄 뉴스 수집 및 AI 요약 시작"):
        with st.spinner("실시간 경제 뉴스 수집 및 AI 브리핑 생성 중..."):
            st.session_state.news_data = get_naver_api_news()
            if st.session_state.news_data:
                db, summary = init_langchain_rag(st.session_state.news_data)
                st.session_state.vector_db = db
                st.session_state.summary = summary
                st.success("AI 브리핑 준비 완료!")
            else:
                st.warning("뉴스를 가져오지 못했습니다. API 키 설정을 확인해 보세요.")

    if st.session_state.summary:
        with st.container():
            st.markdown("---")
            st.write("### 📢 오늘의 경제 브리핑")
            st.info(st.session_state.summary)
            st.markdown("---")

    for idx, news in enumerate(st.session_state.news_data, 1):
        with st.expander(f"{idx}. {news['title']}"):
            st.write(news['content'])
            st.caption(f"[기사 원본 링크]({news['link']})")

with col2:
    st.write("### 🤖 오늘의 경제 뉴스에 대해 물어보세요")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    if user_question := st.chat_input("수집된 뉴스 내용에 대해 궁금한 점을 질문해보세요!"):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
            
        with st.chat_message("assistant"):
            if st.session_state.vector_db is None:
                st.write("먼저 좌측의 '뉴스 수집 및 AI 요약 시작' 버튼을 눌러 데이터를 확보해주세요.")
            else:
                with st.spinner("뉴스 데이터 분석 중..."):
                    docs = st.session_state.vector_db.similarity_search(user_question, k=2)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = ChatGroq(temperature=0.5, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "당신은 친절한 경제 전문가입니다. 제공된 뉴스 내용(Context)을 바탕으로 사용자의 질문에 답변하세요. 만약 뉴스 내용만으로 구체적인 근거를 설명하기 부족하다면, 당신이 가진 경제 지식을 추가하여 '뉴스를 토대로 추론한 배경과 근거'를 친절하게 확장해서 설명해 주세요.\n\n[뉴스 내용]:\n{context}"),
                        ("human", "{question}")
                    ])
                    
                    qa_chain = qa_prompt | llm
                    response = qa_chain.invoke({"context": context, "question": user_question})
                    
                    st.write(response.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})