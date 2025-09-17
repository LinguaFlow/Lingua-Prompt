import time
import re
import os
from flask import Flask, request, jsonify
from typing import List, Dict, Optional, Any, Union
import google.generativeai as genai
app = Flask(__name__)

# Configuration constants
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
    DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY")
    DEFAULT_NUM_EXAMPLES = int(os.getenv("DEFAULT_NUM_EXAMPLES"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES"))  # seconds
    REQUEST_TIMEOUT = 30  # seconds

    # API 안전 설정
    ENABLE_SAFETY_SETTINGS = True
    USE_SIMPLIFIED_PROMPTS = True

    # 폴백 모드 설정
    FALLBACK_ENABLED = True
    FALLBACK_EXAMPLES = {
        "聞く": [
            {"kanji": "聞く", "pos": "動詞", "meaning": "듣다, 묻다", "contexts": ["청각", "질문"]},
            {"kanji": "聴く", "pos": "動詞", "meaning": "주의 깊게 듣다", "contexts": ["감상", "집중"]},
            {"kanji": "効く", "pos": "動詞", "meaning": "효과가 있다", "contexts": ["약효", "작용"]}
        ],
        "きく": [
            {"kanji": "聞く", "pos": "動詞", "meaning": "듣다, 묻다", "contexts": ["청각", "질문"]},
            {"kanji": "聴く", "pos": "動詞", "meaning": "주의 깊게 듣다", "contexts": ["감상", "집중"]},
            {"kanji": "効く", "pos": "動詞", "meaning": "효과가 있다", "contexts": ["약효", "작용"]}
        ]
    }

    # JLPT 레벨 설명
    LEVEL_DESCRIPTIONS = {
        "n5": "Beginner level (N5) - Absolute beginner: Able to understand basic Japanese phrases and sentences when spoken slowly. Can read hiragana, katakana, and about 100 kanji. Vocabulary knowledge includes approximately 800 words covering basic needs and everyday situations. Can introduce oneself and engage in very simple conversations.",

        "n4": "Basic level (N4) - Basic daily expressions: Can understand conversations about everyday topics when spoken slowly. Able to read and write about 300 kanji and comprehend simple passages on familiar topics. Vocabulary knowledge extends to roughly 1,500 words. Can handle basic social interactions and express simple opinions or preferences.",

        "n3": "Intermediate level (N3) - Everyday communication: Can understand coherent Japanese used in daily situations at near-natural speed. Able to read approximately 650 kanji and comprehend newspaper headlines and straightforward articles. Vocabulary knowledge covers about 3,700 words. Can express opinions, describe experiences, and maintain conversations on familiar topics with some fluency.",

        "n2": "Upper intermediate level (N2) - Practical communication: Can understand natural-speed Japanese in a variety of situations. Able to read 1,000+ kanji and comprehend newspapers, general articles, and some specialized content with occasional dictionary use. Vocabulary knowledge extends to roughly 6,000 words. Can communicate effectively in most social and professional contexts, expressing ideas with reasonable clarity and accuracy.",

        "n1": "Advanced level (N1) - Natural, native-like expressions: Can understand Japanese used in a wide range of sophisticated contexts. Able to read 2,000+ kanji and comprehend complex texts including abstract concepts, literary works, and specialized content. Vocabulary knowledge exceeds 10,000 words. Can communicate with nuance and subtlety, demonstrating near-native fluency across various situations and topics.",

        "standard": "Intermediate level (default) - Represents a balanced proficiency level approximately equivalent to N3, suitable for everyday communication and basic professional interactions. Includes knowledge of common vocabulary, grammar patterns, and cultural references sufficient for most general purposes."
    }

    # Detailed grammar instructions by level
    DETAILED_INSTRUCTIONS = {
        "n5": (
            "Create very simple, short sentences using basic subject-object-verb structure. "
            "Use only basic vocabulary (around 800 words) and masu-form present tense. "
            "Include basic particles (は, を, に, で, etc.) and simple question forms. "
            "Use simple time expressions (今日, 明日, etc.) and straightforward counters. "
            "Avoid conjugations beyond the essentials (masu-form, simple negative). "
            "Keep sentences under 8-10 words with minimal compound structures."
        ),

        "n4": (
            "Use basic daily expressions with simple past and present tense forms. "
            "Include te-form for requests, permissions, and ongoing actions. "
            "Incorporate basic conjunctions (そして, でも, から) and time connectors. "
            "Introduce simple potential, imperative, and volitional forms. "
            "Use ~たい for expressing desires and basic compound sentences. "
            "Keep grammar straightforward and beginner-friendly with vocabulary of around 1,500 words. "
            "Limit use of specialized terminology and complex modifier structures."
        ),

        "n3": (
            "Include both casual and polite forms with conditional structures (～たら, ～と, ～ば, ～なら). "
            "Use transitive/intransitive verb pairs appropriately in context. "
            "Incorporate provisional, causative, and passive expressions. "
            "Employ more complex particles (について, によって) and conjunction patterns. "
            "Use everyday vocabulary (around 3,700 words) with some specialized terms. "
            "Include appropriate sentence-ending particles for natural conversation. "
            "Demonstrate proper use of embedded clauses and complex modifiers."
        ),

        "n2": (
            "Employ more complex grammar including honorific and humble expressions. "
            "Use keigo where appropriate (尊敬語, 謙譲語, 丁寧語) in social contexts. "
            "Include advanced conditional forms and complex cause-effect relationships. "
            "Incorporate idiomatic phrasing and expressions for practical contexts. "
            "Use specialized vocabulary (around 6,000 words) relevant to professional settings. "
            "Show proper use of complex modifiers, nominalizers, and sentence patterns. "
            "Demonstrate appropriate register switching based on social context."
        ),

        "n1": (
            "Use advanced, native-like expressions including formal honorifics, "
            "humble language, and situationally-appropriate speech styles. "
            "Incorporate nuanced idioms, proverbs, and culturally-specific references. "
            "Employ sophisticated rhetorical techniques and literary expressions. "
            "Use specialized vocabulary (10,000+ words) with proper field-specific terminology. "
            "Include complex grammatical structures rarely found in textbooks. "
            "Provide cultural depth through appropriate speech patterns and sophisticated expressions. "
            "Demonstrate native-like command of subtle connotations and implications."
        ),

        "standard": (
            "Use general intermediate grammar including plain and polite forms. "
            "Incorporate common conditional structures and modal expressions. "
            "Use moderate vocabulary (approximately 3,000-4,000 words) with some specialized terms. "
            "Balance complexity and clarity in sentence structures. "
            "Include commonly used idiomatic expressions and natural phrasing. "
            "Demonstrate appropriate use of casual and formal speech based on context."
        ),
    }

    HOMONYM_DATABASE = {
        "n5": {
            "あめ": [
                {"kanji": "雨", "pos": "名詞", "meaning": "하늘에서 내리는 물", "contexts": ["날씨", "비가 오다"]},
                {"kanji": "飴", "pos": "名詞", "meaning": "달콤한 과자", "contexts": ["사탕", "간식"]}
            ],
            "はな": [
                {"kanji": "花", "pos": "名詞", "meaning": "식물의 꽃", "contexts": ["꽃다발", "벚꽃"]},
                {"kanji": "鼻", "pos": "名詞", "meaning": "얼굴 중앙의 기관", "contexts": ["냄새 맡다", "코피"]}
            ],
            "はし": [
                {"kanji": "箸", "pos": "名詞", "meaning": "식사용 도구", "contexts": ["젓가락질", "식사"]},
                {"kanji": "橋", "pos": "名詞", "meaning": "강에 건설한 길", "contexts": ["다리 건너기", "교통"]}
            ],
            "きる": [
                {"kanji": "切る", "pos": "動詞", "meaning": "자르다", "contexts": ["종이 자르기", "요리"]},
                {"kanji": "着る", "pos": "動詞", "meaning": "입다", "contexts": ["옷 입기", "착용"]}
            ],
            "いる": [
                {"kanji": "居る", "pos": "動詞", "meaning": "있다, 존재하다", "contexts": ["사람이 있다", "머물다"]},
                {"kanji": "要る", "pos": "動詞", "meaning": "필요하다", "contexts": ["도움이 필요하다", "요구"]}
            ],
            "おじさん": [
                {"kanji": "叔父さん", "pos": "名詞", "meaning": "삼촌", "contexts": ["가족 관계", "친척"]},
                {"kanji": "小父さん", "pos": "名詞", "meaning": "나이 든 남성", "contexts": ["아저씨", "호칭"]}
            ],
            "おばさん": [
                {"kanji": "伯母さん", "pos": "名詞", "meaning": "이모/고모", "contexts": ["가족 관계", "친척"]},
                {"kanji": "小母さん", "pos": "名詞", "meaning": "나이 든 여성", "contexts": ["아주머니", "호칭"]}
            ],
            "ゆき": [
                {"kanji": "雪", "pos": "名詞", "meaning": "눈", "contexts": ["겨울", "눈사람"]},
                {"kanji": "行き", "pos": "名詞", "meaning": "~행, 목적지", "contexts": ["도쿄행", "방향"]}
            ],
            "くも": [
                {"kanji": "雲", "pos": "名詞", "meaning": "하늘에 떠 있는 구름", "contexts": ["날씨", "하늘"]},
                {"kanji": "蜘蛛", "pos": "名詞", "meaning": "거미", "contexts": ["곤충", "거미줄"]}
            ],
            "あつい": [
                {"kanji": "暑い", "pos": "形容詞", "meaning": "기온이 높다", "contexts": ["여름", "더운 날씨"]},
                {"kanji": "熱い", "pos": "形容詞", "meaning": "온도가 높다", "contexts": ["뜨거운 물", "열기"]}
            ],
            "め": [
                {"kanji": "目", "pos": "名詞", "meaning": "시각 기관", "contexts": ["보다", "눈동자"]},
                {"kanji": "芽", "pos": "名詞", "meaning": "식물의 새싹", "contexts": ["새싹", "발아"]}
            ],
            "いま": [
                {"kanji": "今", "pos": "名詞", "meaning": "현재", "contexts": ["지금", "현재 시점"]},
                {"kanji": "居間", "pos": "名詞", "meaning": "거실", "contexts": ["방", "생활공간"]}
            ]
        },
        "n4": {
            "かみ": [
                {"kanji": "紙", "pos": "名詞", "meaning": "문서에 사용하는 재료", "contexts": ["종이", "인쇄"]},
                {"kanji": "髪", "pos": "名詞", "meaning": "머리카락", "contexts": ["헤어스타일", "미용"]},
                {"kanji": "神", "pos": "名詞", "meaning": "신앙의 대상", "contexts": ["종교", "신사"]}
            ],
            "しろ": [
                {"kanji": "白", "pos": "名詞", "meaning": "흰색", "contexts": ["색깔", "순수"]},
                {"kanji": "城", "pos": "名詞", "meaning": "성곽", "contexts": ["역사", "건축물"]}
            ],
            "かう": [
                {"kanji": "買う", "pos": "動詞", "meaning": "구입하다", "contexts": ["쇼핑", "구매"]},
                {"kanji": "飼う", "pos": "動詞", "meaning": "동물을 기르다", "contexts": ["애완동물", "사육"]}
            ],
            "あう": [
                {"kanji": "会う", "pos": "動詞", "meaning": "만나다", "contexts": ["약속", "만남"]},
                {"kanji": "合う", "pos": "動詞", "meaning": "맞다, 적합하다", "contexts": ["어울리다", "조화"]}
            ],
            "とる": [
                {"kanji": "取る", "pos": "動詞", "meaning": "손에 잡다", "contexts": ["집다", "획득"]},
                {"kanji": "撮る", "pos": "動詞", "meaning": "사진을 찍다", "contexts": ["촬영", "기록"]},
                {"kanji": "捕る", "pos": "動詞", "meaning": "잡다", "contexts": ["사냥", "포획"]}
            ],
            "とぶ": [
                {"kanji": "飛ぶ", "pos": "動詞", "meaning": "공중을 이동하다", "contexts": ["비행", "날다"]},
                {"kanji": "跳ぶ", "pos": "動詞", "meaning": "뛰다", "contexts": ["점프", "도약"]}
            ],
            "なく": [
                {"kanji": "鳴く", "pos": "動詞", "meaning": "동물이 소리내다", "contexts": ["새 울음소리", "동물"]},
                {"kanji": "泣く", "pos": "動詞", "meaning": "눈물을 흘리다", "contexts": ["슬프다", "감정"]}
            ],
            "かわ": [
                {"kanji": "川", "pos": "名詞", "meaning": "물의 흐름", "contexts": ["강", "자연"]},
                {"kanji": "皮", "pos": "名詞", "meaning": "피부, 가죽", "contexts": ["동물 가죽", "표면"]}
            ],
            "かた": [
                {"kanji": "方", "pos": "名詞", "meaning": "사람을 높여 부르는 말", "contexts": ["호칭", "존경"]},
                {"kanji": "肩", "pos": "名詞", "meaning": "팔의 뿌리", "contexts": ["어깨", "신체"]}
            ],
            "あがる・あげる": [
                {"kanji": "上げる", "pos": "動詞", "meaning": "올리다", "contexts": ["상승", "높이다"]},
                {"kanji": "揚げる", "pos": "動詞", "meaning": "기름에 튀기다", "contexts": ["요리", "튀김"]},
                {"kanji": "挙げる", "pos": "動詞", "meaning": "예를 들다", "contexts": ["언급", "거론"]}
            ]
        },
        "n3": {
            "かぶ": [
                {"kanji": "株", "pos": "名詞", "meaning": "주식", "contexts": ["투자", "금융"]},
                {"kanji": "蕪", "pos": "名詞", "meaning": "순무", "contexts": ["야채", "농업"]}
            ],
            "きかく": [
                {"kanji": "企画", "pos": "名詞", "meaning": "계획", "contexts": ["프로젝트", "기획"]},
                {"kanji": "規格", "pos": "名詞", "meaning": "표준", "contexts": ["기준", "규격"]}
            ],
            "しじ": [
                {"kanji": "指示", "pos": "名詞", "meaning": "지시", "contexts": ["명령", "안내"]},
                {"kanji": "支持", "pos": "名詞", "meaning": "지지", "contexts": ["응원", "후원"]}
            ],
            "けいき": [
                {"kanji": "景気", "pos": "名詞", "meaning": "경기", "contexts": ["경제 상황", "호황"]},
                {"kanji": "契機", "pos": "名詞", "meaning": "계기", "contexts": ["기회", "전환점"]}
            ],
            "しょうひん": [
                {"kanji": "商品", "pos": "名詞", "meaning": "상품", "contexts": ["판매품", "제품"]},
                {"kanji": "賞品", "pos": "名詞", "meaning": "상품", "contexts": ["경품", "상금"]}
            ],
            "こうざ": [
                {"kanji": "講座", "pos": "名詞", "meaning": "강좌", "contexts": ["강의", "교육"]},
                {"kanji": "口座", "pos": "名詞", "meaning": "계좌", "contexts": ["은행", "금융"]}
            ],
            "げんきん": [
                {"kanji": "現金", "pos": "名詞", "meaning": "현금", "contexts": ["돈", "지불"]},
                {"kanji": "厳禁", "pos": "名詞", "meaning": "엄금", "contexts": ["금지", "규칙"]}
            ]
        },
        "n2": {
            "せいり": [
                {"kanji": "整理", "pos": "名詞", "meaning": "정리", "contexts": ["정돈", "정리정돈"]},
                {"kanji": "生理", "pos": "名詞", "meaning": "생리", "contexts": ["월경", "생리학"]}
            ],
            "いたみ": [
                {"kanji": "痛み", "pos": "名詞", "meaning": "고통", "contexts": ["아픔", "통증"]},
                {"kanji": "傷み", "pos": "名詞", "meaning": "손상", "contexts": ["부패", "상함"]}
            ],
            "いし": [
                {"kanji": "医師", "pos": "名詞", "meaning": "의사", "contexts": ["의료진", "병원"]},
                {"kanji": "石", "pos": "名詞", "meaning": "돌", "contexts": ["바위", "광물"]},
                {"kanji": "意思", "pos": "名詞", "meaning": "의지", "contexts": ["생각", "의도"]}
            ],
            "どうき": [
                {"kanji": "動機", "pos": "名詞", "meaning": "동기", "contexts": ["이유", "목적"]},
                {"kanji": "動悸", "pos": "名詞", "meaning": "동계", "contexts": ["심장박동", "의료"]}
            ],
            "けっかん": [
                {"kanji": "血管", "pos": "名詞", "meaning": "혈관", "contexts": ["의료", "순환기"]},
                {"kanji": "欠陥", "pos": "名詞", "meaning": "결함", "contexts": ["문제", "부족"]}
            ],
            "ほけん": [
                {"kanji": "保健", "pos": "名詞", "meaning": "보건", "contexts": ["건강", "의료"]},
                {"kanji": "保険", "pos": "名詞", "meaning": "보험", "contexts": ["계약", "보장"]}
            ],
            "はい": [
                {"kanji": "肺", "pos": "名詞", "meaning": "폐", "contexts": ["호흡기", "의료"]},
                {"kanji": "灰", "pos": "名詞", "meaning": "재", "contexts": ["연소", "화산재"]}
            ],
            "のう": [
                {"kanji": "脳", "pos": "名詞", "meaning": "뇌", "contexts": ["머리", "의료"]},
                {"kanji": "能", "pos": "名詞", "meaning": "능력", "contexts": ["기능", "재능"]}
            ],
            "せいけい": [
                {"kanji": "整形", "pos": "名詞", "meaning": "성형", "contexts": ["수술", "의료"]},
                {"kanji": "成形", "pos": "名詞", "meaning": "성형", "contexts": ["모양 만들기", "제작"]}
            ],
            "いしょく": [
                {"kanji": "移植", "pos": "名詞", "meaning": "이식", "contexts": ["장기이식", "의료"]},
                {"kanji": "異色", "pos": "名詞", "meaning": "이색", "contexts": ["특이함", "독특함"]}
            ],
            "よむ": [
                {"kanji": "読む", "pos": "動詞", "meaning": "읽다", "contexts": ["독서", "학습"]},
                {"kanji": "詠む", "pos": "動詞", "meaning": "읊다", "contexts": ["시가", "문학"]}
            ],
            "みる": [
                {"kanji": "見る", "pos": "動詞", "meaning": "보다", "contexts": ["시각", "관찰"]},
                {"kanji": "観る", "pos": "動詞", "meaning": "관람하다", "contexts": ["감상", "공연"]},
                {"kanji": "診る", "pos": "動詞", "meaning": "진찰하다", "contexts": ["의료", "진료"]}
            ],
            "きく": [
                {"kanji": "聞く", "pos": "動詞", "meaning": "듣다, 묻다", "contexts": ["청각", "질문"]},
                {"kanji": "聴く", "pos": "動詞", "meaning": "주의 깊게 듣다", "contexts": ["감상", "집중"]},
                {"kanji": "効く", "pos": "動詞", "meaning": "효과가 있다", "contexts": ["약효", "작용"]}
            ],
            "つくる": [
                {"kanji": "作る", "pos": "動詞", "meaning": "만들다", "contexts": ["제작", "창작"]},
                {"kanji": "造る", "pos": "動詞", "meaning": "건조하다", "contexts": ["건축", "건설"]},
                {"kanji": "創る", "pos": "動詞", "meaning": "창조하다", "contexts": ["창작", "예술"]}
            ],
            "あらわす": [
                {"kanji": "表す", "pos": "動詞", "meaning": "표현하다", "contexts": ["감정", "의견"]},
                {"kanji": "現す", "pos": "動詞", "meaning": "나타내다", "contexts": ["출현", "드러내다"]},
                {"kanji": "著す", "pos": "動詞", "meaning": "저술하다", "contexts": ["집필", "출간"]}
            ],
            "さす": [
                {"kanji": "指す", "pos": "動詞", "meaning": "가리키다", "contexts": ["방향", "지시"]},
                {"kanji": "差す", "pos": "動詞", "meaning": "우산을 쓰다", "contexts": ["우산", "햇빛"]},
                {"kanji": "刺す", "pos": "動詞", "meaning": "찌르다", "contexts": ["날카로움", "공격"]},
                {"kanji": "挿す", "pos": "動詞", "meaning": "꽂다", "contexts": ["삽입", "장식"]}
            ],
            "はなす": [
                {"kanji": "話す", "pos": "動詞", "meaning": "말하다", "contexts": ["대화", "소통"]},
                {"kanji": "放す", "pos": "動詞", "meaning": "놓아주다", "contexts": ["해방", "방출"]},
                {"kanji": "離す", "pos": "動詞", "meaning": "떨어뜨리다", "contexts": ["분리", "거리"]}
            ],
            "すすめる": [
                {"kanji": "進める", "pos": "動詞", "meaning": "전진시키다", "contexts": ["진행", "발전"]},
                {"kanji": "勧める", "pos": "動詞", "meaning": "권하다", "contexts": ["추천", "제안"]},
                {"kanji": "薦める", "pos": "動詞", "meaning": "추천하다", "contexts": ["천거", "소개"]}
            ],
            "おさめる": [
                {"kanji": "収める", "pos": "動詞", "meaning": "얻다", "contexts": ["수입", "획득"]},
                {"kanji": "納める", "pos": "動詞", "meaning": "납입하다", "contexts": ["세금", "지불"]},
                {"kanji": "治める", "pos": "動詞", "meaning": "다스리다", "contexts": ["통치", "관리"]},
                {"kanji": "修める", "pos": "動詞", "meaning": "습득하다", "contexts": ["학습", "연마"]}
            ]
        },
        "n1": {
            "しょうにん": [
                {"kanji": "証人", "pos": "名詞", "meaning": "증인", "contexts": ["법정", "증언"]},
                {"kanji": "商人", "pos": "名詞", "meaning": "상인", "contexts": ["장사", "무역"]}
            ],
            "しこう": [
                {"kanji": "施行", "pos": "名詞", "meaning": "시행", "contexts": ["법률", "실시"]},
                {"kanji": "試行", "pos": "名詞", "meaning": "시행", "contexts": ["테스트", "실험"]},
                {"kanji": "志向", "pos": "名詞", "meaning": "지향", "contexts": ["목표", "방향성"]}
            ],
            "ほしょう": [
                {"kanji": "保証", "pos": "名詞", "meaning": "보증", "contexts": ["약속", "담보"]},
                {"kanji": "保障", "pos": "名詞", "meaning": "보장", "contexts": ["보호", "안전"]},
                {"kanji": "補償", "pos": "名詞", "meaning": "보상", "contexts": ["배상", "손해"]}
            ],
            "ふよう": [
                {"kanji": "不要", "pos": "形容動詞", "meaning": "불필요", "contexts": ["필요없음", "무용"]},
                {"kanji": "扶養", "pos": "名詞", "meaning": "부양", "contexts": ["양육", "지원"]}
            ],
            "たいほ": [
                {"kanji": "逮捕", "pos": "名詞", "meaning": "체포", "contexts": ["경찰", "범죄"]},
                {"kanji": "大砲", "pos": "名詞", "meaning": "대포", "contexts": ["무기", "군사"]}
            ],
            "こじん": [
                {"kanji": "個人", "pos": "名詞", "meaning": "개인", "contexts": ["개별", "사적"]},
                {"kanji": "故人", "pos": "名詞", "meaning": "고인", "contexts": ["사망자", "추모"]}
            ],
            "けいじ": [
                {"kanji": "刑事", "pos": "名詞", "meaning": "형사", "contexts": ["경찰", "수사"]},
                {"kanji": "掲示", "pos": "名詞", "meaning": "게시", "contexts": ["공지", "알림"]}
            ],
            "きそ": [
                {"kanji": "起訴", "pos": "名詞", "meaning": "기소", "contexts": ["법정", "소송"]},
                {"kanji": "基礎", "pos": "名詞", "meaning": "기초", "contexts": ["토대", "바탕"]}
            ],
            "こくさい": [
                {"kanji": "国際", "pos": "名詞", "meaning": "국제", "contexts": ["국가간", "세계"]},
                {"kanji": "国債", "pos": "名詞", "meaning": "국채", "contexts": ["국가 부채", "금융"]}
            ],
            "しょめい": [
                {"kanji": "署名", "pos": "名詞", "meaning": "서명", "contexts": ["사인", "계약"]},
                {"kanji": "書名", "pos": "名詞", "meaning": "서명", "contexts": ["책 제목", "도서"]}
            ]
        }
    }

    # Valid JLPT levels
    VALID_LEVELS = ["n5", "n4", "n3", "n2", "n1", "standard"]


class LLMService:
    """
    Enhanced service class for handling Large Language Model interactions with Google's Gemini AI.

    This class provides robust methods to initialize the Gemini API connection and make
    API calls with comprehensive error handling, retry logic, safety setting adjustments,
    and detailed response processing for various failure scenarios.
    """

    @staticmethod
    def initialize_gemini():
        """
        Initialize the Gemini API with the configured API key.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not Config.GEMINI_API_KEY:
            app.logger.error("API key not configured. Set GEMINI_API_KEY environment variable.")
            return False

        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            return True
        except Exception as e:
            app.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            return False

    @staticmethod
    def _create_safety_settings() -> List[Dict[str, Union[str, int]]]:
        """
        Create properly formatted safety settings for Gemini API.

        Returns:
            List of safety setting dictionaries
        """
        try:
            # Method 1: Using enum values directly (preferred)
            return [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        except Exception as e:
            app.logger.warning(f"Failed to create safety settings: {e}")
            return []

    @staticmethod
    def _create_model_with_safety_settings():
        """
        Create Gemini model with proper safety settings.

        Returns:
            GenerativeModel instance with safety settings applied
        """
        try:
            # Method 1: Try with list format (most compatible)
            safety_settings = LLMService._create_safety_settings()
            model = genai.GenerativeModel(
                Config.MODEL_NAME,
                safety_settings=safety_settings
            )
            return model

        except Exception as e:
            app.logger.warning(f"Failed to create model with safety settings: {e}")

            # Method 2: Try with dictionary format (alternative)
            try:
                safety_dict = {
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }

                model = genai.GenerativeModel(
                    Config.MODEL_NAME,
                    safety_settings=safety_dict  # type: ignore[arg-type]
                )
                return model

            except Exception as e2:
                app.logger.warning(f"Alternative safety settings also failed: {e2}")

                # Method 3: No safety settings (fallback)
                app.logger.info("Creating model without custom safety settings")
                return genai.GenerativeModel(Config.MODEL_NAME)

    @staticmethod
    def call_llm(
            prompt: str,
            temperature: float = Config.DEFAULT_TEMPERATURE
    ) -> Optional[str]:
        """
        Make a robust call to the Gemini LLM with comprehensive error handling.

        Args:
            prompt: The input prompt to send to the model
            temperature: Sampling temperature for response generation (0.0-1.0)

        Returns:
            Generated text response or None if all retries failed
        """
        if not Config.GEMINI_API_KEY:
            app.logger.error("API key not configured. Set GEMINI_API_KEY environment variable.")
            return None

        # Initialize Gemini API if not already initialized
        LLMService.initialize_gemini()

        # Retry logic for API call
        retry_count = 0

        while retry_count < Config.MAX_RETRIES:
            try:
                # Create model with safety settings
                model = LLMService._create_model_with_safety_settings()

                # Create generation config - using dictionary as it's more reliable
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "candidate_count": 1,
                }

                # Generate content with error handling
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config  # type: ignore[arg-type]
                )

                # Comprehensive response validation
                if response is None:
                    app.logger.error("Received None response from Gemini API")
                    retry_count += 1
                    continue

                # Check if response has candidates
                if not hasattr(response, 'candidates') or not response.candidates:
                    app.logger.error("No candidates in response")
                    retry_count += 1
                    continue

                # Check finish reason for the first candidate
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason

                app.logger.debug(f"Response finish_reason: {finish_reason}")

                # Handle different finish reasons
                if finish_reason == 1:  # STOP - successful completion
                    try:
                        return response.text
                    except Exception as text_error:
                        app.logger.error(f"Error accessing response.text: {text_error}")
                        # Try alternative text access
                        try:
                            if candidate.content and candidate.content.parts:
                                return candidate.content.parts[0].text
                        except Exception as alt_error:
                            app.logger.error(f"Alternative text access failed: {alt_error}")

                elif finish_reason == 2:  # MAX_TOKENS
                    app.logger.warning("Response truncated due to max tokens limit")
                    try:
                        return response.text
                    except:
                        try:
                            if candidate.content and candidate.content.parts:
                                return candidate.content.parts[0].text
                        except:
                            pass

                elif finish_reason == 3:  # SAFETY
                    app.logger.warning("Response blocked by safety filters")
                    # Try with modified prompt
                    if retry_count < Config.MAX_RETRIES - 1:
                        app.logger.info("Retrying with modified prompt...")
                        # Simplify the prompt to avoid safety issues
                        simplified_prompt = LLMService._simplify_prompt(prompt)
                        if simplified_prompt != prompt:
                            prompt = simplified_prompt
                            retry_count += 1
                            continue

                elif finish_reason == 4:  # RECITATION
                    app.logger.warning("Response blocked due to recitation")

                else:  # OTHER or unknown
                    app.logger.warning(f"Unknown finish_reason: {finish_reason}")

                # If we get here, the response wasn't successful
                retry_count += 1
                if retry_count < Config.MAX_RETRIES:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    app.logger.info(
                        f"Retrying in {wait_time} seconds... (attempt {retry_count + 1}/{Config.MAX_RETRIES})")
                    time.sleep(wait_time)

            except Exception as e:
                error_msg = str(e)
                app.logger.error(f"Gemini API call error: {error_msg}")

                # Handle specific error types
                if "500" in error_msg or "internal error" in error_msg.lower():
                    app.logger.info("Server error detected, waiting longer before retry...")
                    time.sleep(5)
                elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    app.logger.warning("Rate limit or quota exceeded, waiting...")
                    time.sleep(10)

                retry_count += 1
                if retry_count < Config.MAX_RETRIES:
                    time.sleep(2)

        app.logger.error(f"All {Config.MAX_RETRIES} retry attempts failed")
        return None

    @staticmethod
    def _simplify_prompt(original_prompt: str) -> str:
        """
        Simplify prompt to avoid safety filter issues.

        Args:
            original_prompt: Original prompt that may have triggered safety filters

        Returns:
            Simplified version of the prompt
        """
        # Remove potentially problematic sections and simplify
        simplified = original_prompt

        # Remove complex formatting that might confuse safety filters
        simplified = re.sub(r'##[^#]*?##', '', simplified)
        simplified = re.sub(r'```json.*?```', '', simplified, flags=re.DOTALL)

        # Keep only the essential task description
        lines = simplified.split('\n')
        essential_lines = []

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in [
                'analyze', 'japanese', 'word', 'homonym', 'meaning', 'translation'
            ]):
                essential_lines.append(line)

        if essential_lines:
            return '\n'.join(essential_lines[:10])  # Limit to 10 lines
        else:
            return "Analyze the Japanese word and provide its meanings in Korean."

    @staticmethod
    def get_api_status() -> Dict[str, Any]:
        """
        Check the current status of the Gemini API connection.

        Returns:
            Dictionary with API status information
        """
        try:
            # Simple test call
            model = LLMService._create_model_with_safety_settings()
            test_response = model.generate_content(
                "Test",
                generation_config={"max_output_tokens": 10}  # type: ignore[arg-type]
            )

            return {
                "status": "healthy",
                "api_key_configured": bool(Config.GEMINI_API_KEY),
                "model_accessible": True,
                "test_response_received": test_response is not None,
                "safety_settings_applied": True
            }
        except Exception as e:
            return {
                "status": "error",
                "api_key_configured": bool(Config.GEMINI_API_KEY),
                "model_accessible": False,
                "error": str(e),
                "safety_settings_applied": False
            }


class HomonymExampleGenerator:
    """
    Main service class for Japanese homonym detection and example sentence generation.

    This class provides comprehensive functionality for identifying Japanese homonyms
    (words with the same pronunciation but different meanings and kanji) and generating
    educational example sentences for each meaning. It combines a pre-built database
    of JLPT-level homonyms with AI-powered fallback detection.

    Key features:
    - Hybrid search approach: database-first, AI-assisted fallback
    - JLPT level-appropriate content generation
    - Context-aware example sentence creation
    - Kanji usage verification and correction
    - Educational formatting with learning tips

    The class supports all JLPT levels (N5-N1) and provides detailed explanations
    to help learners distinguish between homonym variants through context and kanji usage.
    """

    @staticmethod
    def find_homonym_meanings(
            word: str,
            level: str = Config.DEFAULT_DIFFICULTY
    ) -> List[Dict]:
        """
        Find homonym meanings using database-first approach with AI fallback.

        Searches for Japanese homonyms by first checking the built-in database
        organized by JLPT levels, then falling back to AI-powered detection
        if no matches are found in the database.

        Args:
            word: Japanese word (hiragana, katakana, or kanji)
            level: JLPT level for appropriate difficulty (n5, n4, n3, n2, n1)

        Returns:
            List of dictionaries containing homonym information with keys:
            - kanji: The kanji form of the homonym
            - pos: Part of speech in Japanese
            - meaning: Korean translation/meaning
            - contexts: List of usage contexts in Korean
        """
        # 1. 먼저 데이터베이스에서 찾기
        database_results = HomonymExampleGenerator._find_from_database(word, level)

        if database_results:
            app.logger.info(f"Found homonyms for '{word}' in database: {len(database_results)} meanings")
            return database_results

        # 2. 데이터베이스에 없으면 LLM으로 찾기
        app.logger.info(f"'{word}' not found in database, using LLM fallback")
        return HomonymExampleGenerator._find_from_llm(word, level)

    @staticmethod
    def _find_from_database(word: str, level: str) -> List[Dict]:
        """
        Search for homonyms in the pre-built database organized by JLPT levels.

        Searches through JLPT levels from the specified level down to N5,
        looking for exact matches in both reading (hiragana) and kanji forms.

        Args:
            word: Japanese word to search for
            level: Starting JLPT level for search

        Returns:
            List of homonym dictionaries or empty list if not found
        """
        # 현재 레벨과 하위 레벨들에서 찾기
        level_order = ["n5", "n4", "n3", "n2", "n1"]
        current_level_index = level_order.index(level) if level in level_order else 2  # 기본값: n3

        # 현재 레벨부터 N5까지 역순으로 검색 (하위 레벨 우선)
        search_levels = level_order[:current_level_index + 1]

        for search_level in reversed(search_levels):
            level_data = Config.HOMONYM_DATABASE.get(search_level, {})

            # 정확한 읽기로 찾기
            if word in level_data:
                homonyms = level_data[word]
                app.logger.debug(f"Found '{word}' in level {search_level}")
                return homonyms

            # 한자 형태로도 찾기 (예: "聞く" 입력 시 "きく" 키에서 찾기)
            for reading, homonym_list in level_data.items():
                for homonym in homonym_list:
                    if homonym["kanji"] == word:
                        # 같은 읽기의 다른 한자들 반환 (자기 자신 제외)
                        result = [h for h in homonym_list if h["kanji"] != word]
                        if result:
                            app.logger.debug(f"Found '{word}' as kanji variant in level {search_level}")
                            return result

        return []

    @staticmethod
    def _find_from_llm(word: str, level: str) -> List[Dict]:
        """
        Enhanced LLM-based homonym detection with fallback system.
        """
        level_text = Config.LEVEL_DESCRIPTIONS.get(level, Config.LEVEL_DESCRIPTIONS["standard"])
        database_examples = HomonymExampleGenerator._get_database_examples_for_prompt(level)

        # Check fallback first for known problematic words
        if Config.FALLBACK_ENABLED and word in Config.FALLBACK_EXAMPLES:
            app.logger.info(f"Using fallback data for '{word}'")
            return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]

        # Simplified prompt for better API compatibility
        prompt = f"""
        Find Japanese homonyms for: {word}

        Return JSON format:
        {{
          "homonyms_found": true/false,
          "meanings": [
            {{"kanji": "漢字", "pos": "品詞", "meaning": "한국어 의미", "contexts": ["용례1", "용례2"]}}
          ]
        }}

        Examples: {database_examples[:200]}...

        Only return different kanji with same pronunciation.
        """.strip()

        response = LLMService.call_llm(prompt, temperature=0.1)

        if not response:
            app.logger.error(f"Failed to get homonym meanings for {word}")
            # Final fallback: check if we have partial data
            if word in Config.FALLBACK_EXAMPLES:
                app.logger.info(f"Using emergency fallback for '{word}'")
                return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]
            return []

        try:
            # Try to extract JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # If no JSON found, try fallback
                    if word in Config.FALLBACK_EXAMPLES:
                        app.logger.warning(f"JSON parsing failed, using fallback for '{word}'")
                        return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]
                    return []

            import json
            result = json.loads(json_text)

            if not result.get("homonyms_found", False):
                # Try fallback before giving up
                if word in Config.FALLBACK_EXAMPLES:
                    app.logger.info(f"No homonyms found by AI, using fallback for '{word}'")
                    return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]
                return []

            meanings = result.get("meanings", [])

            # Filter and format results
            filtered_meanings = []
            for meaning in meanings:
                if meaning.get("kanji") != word:
                    filtered_meaning = {
                        "kanji": meaning.get("kanji", ""),
                        "pos": meaning.get("pos", "動詞"),
                        "meaning": meaning.get("meaning", ""),
                        "contexts": meaning.get("contexts", [])
                    }
                    filtered_meanings.append(filtered_meaning)

            if not filtered_meanings and word in Config.FALLBACK_EXAMPLES:
                app.logger.info(f"Filtered results empty, using fallback for '{word}'")
                return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]

            return filtered_meanings

        except Exception as e:
            app.logger.error(f"Failed to parse homonym meanings: {str(e)}")
            # Emergency fallback
            if word in Config.FALLBACK_EXAMPLES:
                app.logger.info(f"Exception occurred, using emergency fallback for '{word}'")
                return [h for h in Config.FALLBACK_EXAMPLES[word] if h["kanji"] != word]
            return []

    @staticmethod
    def _get_database_examples_for_prompt(level: str) -> str:
        """
        Generate example homonyms from database to include in AI prompts.

        Extracts sample homonym pairs from the database at the specified JLPT level
        and lower levels to provide context for the AI model.

        Args:
            level: JLPT level to determine example complexity

        Returns:
            Formatted string with homonym examples for prompt inclusion
        """
        examples = []
        level_order = ["n5", "n4", "n3", "n2", "n1"]
        current_index = level_order.index(level) if level in level_order else 2

        # 현재 레벨과 하위 레벨에서 몇 가지 예시 추출
        for search_level in level_order[:current_index + 1]:
            level_data = Config.HOMONYM_DATABASE.get(search_level, {})
            level_examples = list(level_data.items())[:2]  # 각 레벨에서 2개씩

            for reading, homonyms in level_examples:
                kanji_list = [h["kanji"] for h in homonyms]
                meaning_list = [h["meaning"] for h in homonyms]
                examples.append(f"- {reading}: {', '.join(kanji_list)} ({', '.join(meaning_list)})")

        return "\n".join(examples[:8])  # 최대 8개 예시

    @staticmethod
    def generate_homonym_examples(
            word: str,
            level: str = Config.DEFAULT_DIFFICULTY,
            num_examples_per_meaning: int = 3
    ) -> Dict[str, Any]:
        """
        Generate homonym examples with minimal JSON response format.
        Removes source and word fields from response.
        """
        meanings = HomonymExampleGenerator.find_homonym_meanings(word, level)

        if not meanings:
            app.logger.warning(f"No homonym meanings found for '{word}' at level {level}")
            return {
                "found": False,
                "error": f"'{word}'에 대한 동음이의어 정보를 찾을 수 없습니다. 데이터베이스와 AI 검색 모두에서 결과가 없습니다."
            }

        result = {
            "found": True,
            "meanings": []
            # source와 word 필드 제거
        }

        # Get level-specific instruction components
        level_text = Config.LEVEL_DESCRIPTIONS.get(level, Config.LEVEL_DESCRIPTIONS["standard"])
        instruction_detail = Config.DETAILED_INSTRUCTIONS.get(level, Config.DETAILED_INSTRUCTIONS["standard"])

        # Generate examples for each meaning
        for meaning_data in meanings:
            # Convert Japanese pos to Korean
            korean_pos = HomonymExampleGenerator._convert_pos_to_korean(meaning_data["pos"])

            meaning_result = {
                "kanji": meaning_data["kanji"],
                "pos": korean_pos,
                "meaning": meaning_data["meaning"],
                "contexts": meaning_data.get("contexts", []),
                "examples": []
            }

            # Build prompt for this specific meaning
            prompt = HomonymExampleGenerator._build_homonym_example_prompt(
                word,
                meaning_data,
                level_text,
                instruction_detail,
                num_examples_per_meaning
            )

            # Call LLM to generate examples
            response = LLMService.call_llm(prompt)

            if response:
                # Parse examples from response
                examples = HomonymExampleGenerator._parse_examples(response, word, meaning_data["kanji"])

                # Clean examples - remove contains_kanji field
                cleaned_examples = []
                for example in examples:
                    cleaned_example = {
                        "japanese": example["japanese"],
                        "korean": example["korean"],
                        "explanation": example["explanation"]
                    }
                    cleaned_examples.append(cleaned_example)

                # 예시가 3개보다 적으면 기본 예시로 채우기
                while len(cleaned_examples) < 3:
                    app.logger.warning(
                        f"Less than 3 examples generated for {word} ({meaning_data['kanji']}). Adding enhanced placeholder example.")

                    contexts = meaning_data.get("contexts", ["일반적인 사용"])
                    context_example = contexts[len(cleaned_examples) % len(contexts)] if contexts else "일반적인 사용"

                    cleaned_examples.append({
                        "japanese": f"{meaning_data['kanji']}를 사용한 예문입니다.",
                        "korean": f"{meaning_data['meaning']}의 예문입니다.",
                        "explanation": f"이 예문은 '{word}'가 '{meaning_data['meaning']}'라는 의미로 사용된 {context_example} 상황의 예입니다."
                    })

                # 예시가 3개보다 많으면 3개로 제한
                if len(cleaned_examples) > 3:
                    cleaned_examples = cleaned_examples[:3]

                meaning_result["examples"] = cleaned_examples
            else:
                # LLM 응답이 없는 경우 기본 예시 3개 생성
                contexts = meaning_data.get("contexts", ["일반적인 사용", "기본 상황", "예시 상황"])
                meaning_result["examples"] = []

                for i in range(3):
                    context = contexts[i] if i < len(contexts) else f"상황 {i + 1}"
                    meaning_result["examples"].append({
                        "japanese": f"{meaning_data['kanji']}에 관한 {context}의 예문입니다.",
                        "korean": f"{meaning_data['meaning']}에 관한 {context}의 예문입니다.",
                        "explanation": f"이 예문은 '{word}'가 '{meaning_data['meaning']}'라는 의미로 {context}에서 사용된 예입니다."
                    })

            result["meanings"].append(meaning_result)

        return result

    def handle_no_homonyms_case(word: str) -> Dict[str, Any]:
        """
        동음이의어가 없는 경우 응답 형식 (source, word 필드 제거)
        """
        return {
            "found": False,
            "error": f"'{word}'에 대한 동음이의어 정보를 찾을 수 없습니다. 데이터베이스와 AI 검색 모두에서 결과가 없습니다."
            # word와 source 필드 제거
        }

    @staticmethod
    def _convert_pos_to_korean(japanese_pos: str) -> str:
        """
        Convert Japanese part-of-speech to Korean.

        Args:
            japanese_pos: Japanese part of speech (e.g., "動詞", "名詞")

        Returns:
            Korean part of speech (e.g., "동사", "명사")
        """
        pos_conversion = {
            "名詞": "명사",
            "動詞": "동사",
            "形容詞": "형용사",
            "形容動詞": "형용동사",
            "副詞": "부사",
            "助詞": "조사",
            "助動詞": "조동사",
            "連体詞": "관형사",
            "接続詞": "접속사",
            "感動詞": "감탄사",
            "代名詞": "대명사",
            "数詞": "수사",
            "接頭詞": "접두사",
            "接尾詞": "접미사"
        }

        return pos_conversion.get(japanese_pos, japanese_pos)  # 변환할 수 없으면 원본 반환

    @staticmethod
    def _build_homonym_example_prompt(
            word: str,
            meaning_data: Dict,
            level_text: str,
            instruction_detail: str,
            num_examples: int = 3
    ) -> str:
        """
        Build detailed prompt for AI to generate homonym-specific example sentences.

        Creates a comprehensive prompt that includes context information,
        other homonyms for comparison, and specific requirements for generating
        educational example sentences that clearly distinguish the target homonym.

        Args:
            word: The pronunciation/reading of the homonym
            meaning_data: Dictionary with kanji, meaning, and context info
            level_text: JLPT level description
            instruction_detail: Level-specific grammar instructions
            num_examples: Number of examples to generate (always 3)

        Returns:
            Formatted prompt string for the AI model
        """
        # 컨텍스트 정보 준비
        contexts = meaning_data.get("contexts", [])
        context_info = "Example contexts: " + ", ".join(contexts) if contexts else ""

        # 같은 읽기의 다른 동음이의어들 찾기 (구별을 위해)
        other_homonyms = HomonymExampleGenerator._get_other_homonyms_info(word, meaning_data["kanji"])

        return f"""
        # Japanese Homonym Example Generator

        ## Role
        You are an experienced Japanese language teacher creating clear example sentences for Japanese homonyms that distinguish different meanings based on context and kanji usage.

        ## Target Homonym Information
        - Pronunciation: "{word}" 
        - Target Kanji: {meaning_data["kanji"]} (MUST use this specific kanji form)
        - Part of speech: {meaning_data["pos"]}
        - Meaning: {meaning_data["meaning"]}
        - {context_info}

        ## Other Homonyms to Distinguish From
        {other_homonyms}

        ## Critical Requirements
        - You MUST generate EXACTLY 3 example sentences. No more, no fewer.
        - Each example must clearly show the kanji "{meaning_data["kanji"]}" being used with the meaning "{meaning_data["meaning"]}"
        - Use {level_text} Japanese appropriate for the learner's level
        - {instruction_detail}
        - EVERY sentence must be COMPLETE and NATURAL. Never leave sentences unfinished.
        - Each example should CLEARLY distinguish this kanji form from other homonym kanji forms listed above
        - ALWAYS include the kanji {meaning_data["kanji"]} in your examples - do not use hiragana only
        - The examples should help learners understand the contextual differences between homonyms

        ## Output Format
        For each example, strictly follow this exact format with no deviation:

        1. Context: [Brief context in English - 1-2 sentences maximum]
        Japanese: [Complete natural Japanese sentence using {meaning_data["kanji"]} with the specified meaning]
        Korean: [Complete natural Korean translation]
        Explanation: [Short explanation in Korean of how this example shows the specific meaning of {meaning_data["kanji"]} and how it differs from other homonyms]

        2. Context: [Brief context in English - 1-2 sentences maximum]  
        Japanese: [Complete natural Japanese sentence using {meaning_data["kanji"]} with the specified meaning]
        Korean: [Complete natural Korean translation]
        Explanation: [Short explanation in Korean of how this example shows the specific meaning of {meaning_data["kanji"]} and how it differs from other homonyms]

        3. Context: [Brief context in English - 1-2 sentences maximum]
        Japanese: [Complete natural Japanese sentence using {meaning_data["kanji"]} with the specified meaning]
        Korean: [Complete natural Korean translation]
        Explanation: [Short explanation in Korean of how this example shows the specific meaning of {meaning_data["kanji"]} and how it differs from other homonyms]

        ## Important Notes
        - You MUST provide EXACTLY 3 examples as numbered above.
        - Make each example very DIFFERENT from the others to show various contexts from: {', '.join(contexts)}
        - ALWAYS use the kanji {meaning_data["kanji"]} in your examples (not hiragana only)
        - Focus on making examples that clearly distinguish {meaning_data["kanji"]} from other homonym forms
        - DO NOT include any unnecessary explanations between examples
        - The Japanese sentences must contain the actual kanji {meaning_data["kanji"]}, not just hiragana
        """.strip()

    @staticmethod
    def _get_other_homonyms_info(pronunciation: str, target_kanji: str) -> str:
        """
        Retrieve information about other homonyms with the same pronunciation.

        Searches the database for other kanji variants that share the same
        pronunciation to provide context for distinguishing homonyms in examples.

        Args:
            pronunciation: The shared pronunciation/reading
            target_kanji: The specific kanji to exclude from results

        Returns:
            Formatted string listing other homonyms for comparison
        """
        other_info = []

        # 모든 레벨에서 같은 발음 검색
        for level_name, level_data in Config.HOMONYM_DATABASE.items():
            for reading, homonyms in level_data.items():
                # 같은 읽기이고 target_kanji가 포함된 경우
                kanji_list = [h["kanji"] for h in homonyms]
                if target_kanji in kanji_list:
                    # 다른 동음이의어들 정보 수집
                    for homonym in homonyms:
                        if homonym["kanji"] != target_kanji:
                            other_info.append(f"- {homonym['kanji']}: {homonym['meaning']} ({homonym['pos']})")
                    break

            if other_info:  # 찾았으면 중단
                break

        if other_info:
            return "Other homonyms with the same pronunciation:\n" + "\n".join(other_info)
        else:
            return "No other known homonyms found in database."

    @staticmethod
    def _parse_examples(
            response_text: str,
            word: str,
            kanji: str
    ) -> List[Dict[str, str]]:
        """
        Parse AI-generated example sentences with simplified output format.
        No longer tracks contains_kanji field.
        """
        examples = []

        app.logger.debug("Response from LLM (first 500 chars): " +
                         (response_text[:500] + "..." if len(response_text) > 500 else response_text))

        # Pattern to match examples
        pattern = r'(?:\d+\.\s*)?Context:\s*(.*?)\s*Japanese:\s*(.*?)\s*Korean:\s*(.*?)\s*Explanation:\s*(.*?)(?=\s*(?:\d+\.\s*)?Context:|\s*$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            app.logger.debug(f"Found {len(matches)} examples with the main pattern")
            for match in matches:
                context, japanese, korean, explanation = match

                # 불완전한 문장 확인
                if japanese.strip().endswith('...') or japanese.strip().endswith('…'):
                    app.logger.debug(f"Skipping incomplete sentence: {japanese}")
                    continue

                if len(japanese.strip()) < 10:
                    app.logger.debug(f"Skipping too short sentence: {japanese}")
                    continue

                # Clean text
                japanese = japanese.strip()
                korean = korean.strip()

                # 설명에서 영어 제거: 줄바꿈 또는 영어 문구 시작 부분까지만 유지
                explanation = explanation.strip()
                # 줄바꿈으로 분리
                if '\n' in explanation:
                    explanation = explanation.split('\n')[0].strip()
                # 또는 다음 Context 시작 부분으로 분리
                if 'Context:' in explanation:
                    explanation = explanation.split('Context:')[0].strip()

                # 한자 포함 여부 체크 (로깅용)
                contains_kanji = kanji in japanese
                if not contains_kanji:
                    app.logger.warning(f"Example does not contain expected kanji '{kanji}': {japanese}")

                # contains_kanji 필드 제거된 형태로 저장
                examples.append({
                    "japanese": japanese,
                    "korean": korean,
                    "explanation": explanation
                    # contains_kanji 필드 제거
                })

        # 한자가 포함된 예문이 없는 경우 처리 (내부 로직용)
        kanji_examples_count = sum(1 for ex in examples if kanji in ex["japanese"])

        if kanji_examples_count == 0 and examples:
            app.logger.warning(f"No examples contain kanji '{kanji}', creating fallback example")
            # 첫 번째 예문을 한자 포함 예문으로 수정
            if examples:
                examples[0]["japanese"] = examples[0]["japanese"].replace(word, kanji)
                examples[0]["explanation"] += f" (한자 {kanji} 사용)"

        return examples

    @staticmethod
    def format_examples_output(homonym_data: Dict[str, Any]) -> str:
        """
        Format homonym example data into a readable output string with visual indicators.

        Creates a well-formatted, educational output string that includes
        homonym examples, context information, visual indicators for kanji usage,
        and learning tips to help students distinguish between homonyms.

        Args:
            homonym_data: Dictionary containing homonym information and examples
                Expected keys: word, found, source, meanings, error

        Returns:
            Formatted string with examples, explanations, and learning guidance
            Including emoji indicators, context info, and educational tips
        """
        if not homonym_data.get("found", False):
            return homonym_data.get("error", "동음이의어 정보를 찾을 수 없습니다.")

        source_info = ""
        if homonym_data.get("source") == "database":
            source_info = " (데이터베이스)"
        elif homonym_data.get("source") == "llm":
            source_info = " (AI 생성)"

        result = f"📝 「{homonym_data['word']}」의 동음이의어 예문{source_info}\n\n"

        for meaning_idx, meaning in enumerate(homonym_data.get("meanings", []), 1):
            # 컨텍스트 정보 추가
            contexts = meaning.get("contexts", [])
            context_info = f" - 사용 맥락: {', '.join(contexts)}" if contexts else ""

            result += f"## {meaning_idx}. {meaning['kanji']} - {meaning['meaning']} ({meaning['pos']})\n"
            result += f"{context_info}\n\n"

            for ex_idx, example in enumerate(meaning.get("examples", []), 1):
                kanji_indicator = "✅" if example.get("contains_kanji", False) else "⚠️"
                result += f"### 예문 {ex_idx} {kanji_indicator}\n"
                result += f"🇯🇵 일본어: {example['japanese']}\n"
                result += f"🇰🇷 한국어: {example['korean']}\n"
                result += f"💡 설명: {example['explanation']}\n\n"

            result += "---\n\n"

        # 학습 팁 추가
        result += "## 📚 학습 팁\n"
        result += "✅ = 한자가 포함된 예문 (권장)\n"
        result += "⚠️ = 한자가 없는 예문 (주의 필요)\n\n"
        result += "💡 동음이의어는 문맥과 한자를 통해 구별할 수 있습니다.\n"

        return result


@app.route('/api/examples', methods=['POST'])
def api_examples():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON 데이터가 필요합니다."}), 400

        word = data.get('word')
        level = data.get('level', Config.DEFAULT_DIFFICULTY)
        format_type = data.get('format', 'simple')  # 'simple', 'with_hiragana', 'with_context'
        homonym_mode = data.get('homonym_mode', False)  # 동음이의어 모드 여부

        if not word:
            return jsonify({"error": "단어가 필요합니다."}), 400

        if level.lower() not in Config.VALID_LEVELS:
            level = Config.DEFAULT_DIFFICULTY

        # 동음이의어 모드인 경우, HomonymExampleGenerator 사용
        if homonym_mode:
            examples = HomonymExampleGenerator.generate_homonym_examples(word, level.lower())
            return jsonify(examples)

        # 일반 예문 생성 모드
        else:
            # 일반 단어의 예문 생성
            examples = generate_word_examples(word, level.lower())

            # 응답 형식에 따라 다른 출력 구성
            if format_type == 'simple':
                # 가장 기본 형식: 일본어와 한국어만 포함
                examples_clean = [
                    {
                        "japanese_example": ex.get("japanese", "").strip(),
                        "korean_translation": ex.get("korean", "").strip()
                    }
                    for ex in examples
                ]
            elif format_type == 'with_context':
                # 컨텍스트 포함 형식
                examples_clean = examples
            elif format_type == 'with_hiragana':
                # 히라가나 독음 추가 (이 기능은 향후 구현 필요)
                examples_clean = [
                    {
                        "japanese_example": ex.get("japanese", "").strip(),
                        "korean_translation": ex.get("korean", "").strip()
                    }
                    for ex in examples
                ]
            else:
                # 기본 형식
                examples_clean = [
                    {
                        "japanese_example": ex.get("japanese", "").strip(),
                        "korean_translation": ex.get("korean", "").strip()
                    }
                    for ex in examples
                ]

            # Return examples
            return jsonify({"examples": examples_clean})

    except Exception as e:
        app.logger.error(f"Example generation error: {str(e)}")
        return jsonify({"error": f"예문 생성 중 오류가 발생했습니다: {str(e)}"}), 500


def generate_word_examples(word: str, level: str = Config.DEFAULT_DIFFICULTY) -> List[Dict]:
    """
    Generate example sentences for general Japanese words (non-homonym mode).

    This function creates educational example sentences that demonstrate the usage
    of a Japanese word in various contexts appropriate for the specified JLPT level.
    Unlike homonym generation, this focuses on showing different grammatical and
    contextual applications of a single word meaning.

    Args:
        word: Japanese word to generate examples for
        level: JLPT level (n5, n4, n3, n2, n1, or standard)

    Returns:
        List of dictionaries containing:
        - japanese: Example sentence in Japanese
        - korean: Korean translation
        - explanation: Korean explanation of usage

    The function uses level-appropriate vocabulary, grammar structures, and
    sentence complexity to match the learner's proficiency level.
    """
    level_text = Config.LEVEL_DESCRIPTIONS.get(level, Config.LEVEL_DESCRIPTIONS["standard"])
    instruction_detail = Config.DETAILED_INSTRUCTIONS.get(level, Config.DETAILED_INSTRUCTIONS["standard"])

    prompt = f"""
    # Japanese Example Sentence Generator

    ## Role
    You are an experienced Japanese language teacher creating clear example sentences for Japanese language learners.

    ## Target Word
    "{word}"

    ## Requirements
    - Generate EXACTLY 5 example sentences that use "{word}" in a natural context
    - Use {level_text} Japanese appropriate for the learner's level
    - {instruction_detail}
    - EVERY sentence must be COMPLETE and NATURAL. Never leave sentences unfinished.
    - Each example should show a different usage context for the word
    - Try to include the word in different grammatical positions

    ## Output Format
    For each example, strictly follow this exact format with no deviation:

    1. Japanese: [Complete natural Japanese sentence that uses the word]
    Korean: [Complete natural Korean translation]
    Explanation: [Short explanation in Korean of how the word is used]

    2. Japanese: [Complete natural Japanese sentence that uses the word]
    Korean: [Complete natural Korean translation]
    Explanation: [Short explanation in Korean of how the word is used]

    ## Important Notes
    - Make each example very DIFFERENT from the others to show various contexts
    - Focus on making examples that would help JLPT learners understand the word's usage
    - Examples should be memorable and useful for test preparation
    - DO NOT include any unnecessary explanations between examples
    """.strip()

    response = LLMService.call_llm(prompt)

    if not response:
        app.logger.error(f"Failed to generate examples for {word}")
        return [{
            "japanese": "例文の生成に失敗しました。",
            "korean": "예문 생성에 실패했습니다.",
            "explanation": "API 응답을 받지 못했습니다."
        }]

    # Pattern to match examples
    pattern = r'(?:\d+\.\s*)?Japanese:\s*(.*?)\s*Korean:\s*(.*?)\s*Explanation:\s*(.*?)(?=\s*(?:\d+\.\s*)?Japanese:|\s*$)'
    matches = re.findall(pattern, response, re.DOTALL)

    examples = []
    if matches:
        app.logger.debug(f"Found {len(matches)} examples with the main pattern")
        for match in matches:
            japanese, korean, explanation = match

            # Clean text
            japanese = japanese.strip()
            korean = korean.strip()
            explanation = explanation.strip()

            examples.append({
                "japanese": japanese,
                "korean": korean,
                "explanation": explanation
            })

    # If no examples were found, create a default error example
    if not examples:
        examples.append({
            "japanese": "例文の生成に失敗しました。",
            "korean": "예문 생성에 실패했습니다.",
            "explanation": "예문 형식에 맞는 결과를 얻지 못했습니다."
        })

    return examples


if __name__ == '__main__':
    # API 키 확인
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and Config.GEMINI_API_KEY == "AIzaSyBq_mtJUx0FJCmfv1uvruNBwpmgVR0vaNU":
        print("警告: API 키가 샘플 키를 사용하고 있습니다. 실제 사용에는 본인의 API 키로 교체하세요.")

    # Flask 애플리케이션 실행
    port = int(os.environ.get("PORT", 3001))
    print(f"\n🚀 일본어 동음이의어 학습기를 시작합니다: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
