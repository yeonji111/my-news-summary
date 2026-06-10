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

# 🔥 [필수 설정] 발급받은 Groq API 키를 여기에 붙여넣으세요.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "여기에_본인의_Groq_키를_넣으세요")

st.title("📰 네이버 실시간 경제 헤드라인 뉴스 요약 봇")
st.subheader("3년 차 SM 개발자의 일주일 완성 사이드 프로젝트 (완성본)")

# 세션 상태(Session State) 초기화
if "news_data" not in st.session_state:
    st.session_state.news_data = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ----------------------------------------------------
# 1단계: 차단 걱정 없는 네이버 경제 헤드라인 크롤링 (이중 그물망 세팅)
# ----------------------------------------------------
def get_naver_headline_news():
    # 네이버 뉴스 경제 섹션 메인 홈 URL
    url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive"
    }
    news_list = []
    
    try:
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # [그물망 1] 헤드라인 및 메인 구역의 모든 <a> 태그를 다 추출합니다.
                links = soup.find_all("a")
                
                for a in links:
                    link = a.get("href", "")
                    title = a.text.strip()
                    
                    # 제목이 너무 짧거나 주소에 'article'과 경제코드 'sid1=101'이 없는 것은 버립니다.
                    if len(title) > 10 and "article" in link and "sid1=101" in link:
                        # 중복 수집 방지
                        if any(n["title"] == title for n in news_list):
                            continue
                            
                        # 본문 긁어오기
                        content = get_news_content_headline(link, session, headers)
                        
                        # 본문을 정상적으로 가져온 경우에만 추가
                        if content and "실패" not in content:
                            news_list.append({
                                "office": "네이버 경제 헤드라인",
                                "title": title,
                                "link": link,
                                "content": content
                            })
                            
                    # 풍부한 질의응답을 위해 헤드라인급 뉴스 20개 모이면 수집 종료
                    if len(news_list) >= 20:
                        break
                        
                # [그물망 2] 만약 그물망 1에서 레이아웃 꼬임으로 안 잡혔을 때를 대비한 백업 선택자
                if len(news_list) == 0:
                    backup_elements = soup.select(".sh_head_title a, .sh_text_headline a, .cluster_text_headline a")
                    for element in backup_elements:
                        link = element.get("href", "")
                        title = element.text.strip()
                        if link and len(title) > 10:
                            content = get_news_content_headline(link, session, headers)
                            news_list.append({
                                "office": "네이버 경제 헤드라인",
                                "title": title,
                                "link": link,
                                "content": content
                            })
                            if len(news_list) >= 20:
                                break
            else:
                st.error(f"네이버 서버 연결 실패 (상태 코드: {response.status_code})")
    except Exception as e:
        st.error(f"크롤링 엔진 에러: {e}")
        
    return news_list

def get_news_content_headline(url, session, headers):
    try:
        res = session.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            # 네이버 기사 본문 고유 영역 ID
            content_area = soup.find(id="dic_area")
            if content_area:
                return content_area.text.strip()
    except:
        pass
    return "본문 내용을 불러오는 데 실패했습니다."

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
        
    with st.spinner("한국어 AI 문맥 분석 모델 가동 중... (데이터셋 임베딩 완료)"):
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vector_db = FAISS.from_documents(documents, embeddings)
        
    all_texts = "\n\n".join([f"[{n['office']}] {n['title']}: {n['content'][:400]}..." for n in news_list])
    
    llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 최고의 경제 브리핑 전문가입니다. 제공된 20개의 핵심 헤드라인 뉴스 데이터를 종합적으로 분석하여, 오늘 가장 뜨거운 경제 이슈 3가지를 선정해 가독성 높게 브리핑해 주세요.
        
        [작성 규칙]:
        1. 각 이슈는 관련된 이모지와 함께 **굵은 제목**으로 보기 좋게 시작하세요.
        2. 각 이슈 아래에는 불릿 포인트(-)를 사용하여 핵심 인과관계와 동향을 2~3줄로 요약하세요.
        3. 마크다운 형식을 적극 활용하여 깔끔하게 출력하세요.
        """)
    ])
    
    chain = summary_prompt | llm
    summary_result = chain.invoke({"input": all_texts})
    
    return vector_db, summary_result.content

# ----------------------------------------------------
# 화면 레이아웃 (좌우 반반 구조)
# ----------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📌 수집된 경제 헤드라인 & AI 브리핑")
    if st.button("🔄 뉴스 수집 및 AI 요약 시작"):
        with st.spinner("네이버 경제 홈에서 실시간 헤드라인 기사 20개 분석 중..."):
            st.session_state.news_data = get_naver_headline_news()
            
            # 🔥 [버그 수정] 과거 API 경고 문구 조건문 정비
            if len(st.session_state.news_data) > 0:
                db, summary = init_langchain_rag(st.session_state.news_data)
                st.session_state.vector_db = db
                st.session_state.summary = summary
                st.success("AI 경제 브리핑 가동 완료!")
            else:
                st.warning("네이버에서 경제 헤드라인 뉴스를 추출하지 못했습니다. 잠시 후 다시 시도해 주세요.")

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
            
    if user_question := st.chat_input("수집된 20개의 헤드라인 뉴스에 대해 무엇이든 질문해 보세요!"):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
            
        with st.chat_message("assistant"):
            if st.session_state.vector_db is None:
                st.write("먼저 좌측의 '뉴스 수집 및 AI 요약 시작' 버튼을 눌러 데이터를 확보해 주세요.")
            else:
                with st.spinner("뉴스 데이터 딥서치 중..."):
                    # 풍부한 답변을 위해 검색 개수(k)를 4개로 확장
                    docs = st.session_state.vector_db.similarity_search(user_question, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "당신은 해박한 경제 지식을 가진 친절한 자산운용가입니다. 제공된 뉴스 내용(Context)을 1순위로 참고하여 답변하되, 질문에 깊이 있는 답변을 주기 위해 당신의 전문 경제 지식을 융합하여 '원인-배경-전망'을 불릿 포인트를 섞어 아주 상세하게 설명해 주세요.\n\n[참고 뉴스 내용]:\n{context}"),
                        ("human", "{question}")
                    ])
                    
                    qa_chain = qa_prompt | llm
                    response = qa_chain.invoke({"context": context, "question": user_question})
                    
                    st.write(response.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})