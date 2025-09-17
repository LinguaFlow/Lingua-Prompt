import os
import time
import re
from flask import Flask, request, jsonify
from typing import List, Dict, Optional
import google.generativeai as genai
from flask.cli import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

load_dotenv()
app = Flask(__name__)


# Configuration constants
class Config:
    """Application configuration settings."""
    # Get API key from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
    DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY")
    DEFAULT_NUM_EXAMPLES = int(os.getenv("DEFAULT_NUM_EXAMPLES"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES")) # seconds

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

    # Usage variation hints by level
    USAGE_VARIATIONS = {
        "n5": (
            "Focus on clear, everyday usage only. "
            "Use simple sentences in common situations like self-introductions, ordering food, "
            "asking directions, and basic shopping interactions. "
            "Limit contexts to classroom, home, and simple travel scenarios. "
            "Avoid slang, colloquialisms, and regional variations. "
            "Present information using only the most fundamental grammatical patterns."
        ),

        "n4": (
            "Show basic usage in common daily situations. "
            "Include simple past experiences, future plans, and basic opinions. "
            "Demonstrate usage in everyday contexts like shopping, dining, travel, and basic workplace interactions. "
            "Introduce simple expressions of preference, ability, and necessity. "
            "Keep examples concrete and related to personal daily routines and needs. "
            "Use straightforward question-answer patterns typical of beginner conversations."
        ),

        "n3": (
            "Mix casual and polite speech in everyday contexts. "
            "Show appropriate switching between speech styles with friends versus acquaintances. "
            "Include usage examples for social gatherings, workplace situations, and service encounters. "
            "Demonstrate expressing opinions, making requests, giving reasons, and discussing plans. "
            "Show proper use in both spoken and written contexts (emails, text messages, etc.). "
            "Include examples of natural conversation flow with appropriate interjections and responses."
        ),

        "n2": (
            "Show usage in a variety of social settings and registers. "
            "Demonstrate appropriate language for business meetings, customer service, and formal occasions. "
            "Include examples of persuasive, descriptive, and explanatory language. "
            "Show proper keigo usage in hierarchical relationships and customer interactions. "
            "Incorporate examples from mass media, academic contexts, and professional settings. "
            "Demonstrate appropriately indirect expressions for requests, refusals, and criticism. "
            "Include examples of situation-specific speech patterns and expressions."
        ),

        "n1": (
            "Demonstrate formal, informal, regional, and professional contexts where relevant. "
            "Include examples of highly formal speech for ceremonial occasions and important business. "
            "Show nuanced differences in expression based on age, gender, social status, and regional background. "
            "Incorporate examples from literature, news media, academic writing, and specialized fields. "
            "Demonstrate subtle differences in nuance between similar expressions. "
            "Show appropriate usage in sensitive situations requiring tact and cultural awareness. "
            "Include examples of rhetorical techniques for persuasion, humor, and emotional impact. "
            "Demonstrate understanding of historical language changes and contemporary trends."
        ),

        "standard": (
            "Use typical everyday polite and casual speech. "
            "Balance formality appropriate to common social and professional situations. "
            "Include examples from daily conversations, routine work interactions, and social media. "
            "Show moderate level of politeness differentiation based on context. "
            "Demonstrate natural conversational patterns with appropriate back-channeling. "
            "Include examples relevant to both spoken and written communication."
        ),
    }

    # Valid JLPT levels
    VALID_LEVELS = ["n5", "n4", "n3", "n2", "n1", "standard"]

    # Verb conjugation descriptions in English (new addition)
    VERB_CONJUGATION_DESCRIPTIONS = {
        "辞書形": "Dictionary form (basic form)",
        "ます形": "Masu form (polite present affirmative)",
        "て形": "Te-form (connecting form, used for requests, ongoing actions)",
        "た形": "Ta-form (past tense)",
        "否定形": "Negative form (present, plain)",
        "ない形": "Nai-form (negative form, base for other negatives)",
        "なかった形": "Nakatta-form (past negative)",
        "命令形": "Imperative form (commands)",
        "禁止形": "Prohibitive form (negative commands)",
        "意向形": "Volitional form (expressing intention)",
        "条件形": "Conditional form (if...then constructions)",
        "可能形": "Potential form (ability to do something)",
        "受身形": "Passive form (being acted upon)",
        "使役形": "Causative form (making/letting someone do something)",
        "使役受身形": "Causative-passive form (being made to do something)",
        "仮定形": "Hypothetical form (if/when scenarios)",
        "敬語": "Honorific form (showing respect)",
        "謙譲語": "Humble form (showing humility for own actions)",
        "丁寧語": "Polite form (general politeness)",
    }

    # Korean translation style guidelines (new addition)
    KOREAN_TRANSLATION_GUIDELINES = {
        "n5": (
            "가장 기본적인 구문과 단어를 사용하여 번역하세요. "
            "자연스러운 한국어로 번역하되, 단순하고 직접적인 표현을 사용하세요. "
            "일상 생활에서 자주 쓰이는 기본적인 표현으로 번역하세요. "
            "한국어 문장을 짧고 명확하게 유지하세요."
        ),

        "n4": (
            "자연스러운 한국어로 번역하되, 일본어 문장 구조를 정확히 반영하세요. "
            "일상적인 표현과 자연스러운 대화체로 번역하세요. "
            "한국인이 실제로 사용하는 자연스러운 표현을 사용하세요. "
            "일본어의 존댓말과 반말 구분을 적절히 번역하세요."
        ),

        "n3": (
            "자연스러운 한국어 표현으로 번역하되, 문맥에 맞게 의역하세요. "
            "직역보다 의미 전달에 중점을 두어 번역하세요. "
            "한국어 고유의 관용적 표현을 적절히 활용하세요. "
            "대화 상황에 맞는 적절한 어조와 말투를 사용하세요."
        ),

        "n2": (
            "일본어 뉘앙스를 살리면서 자연스러운 한국어로 번역하세요. "
            "한국어 관용구와 자연스러운 표현을 적극 활용하세요. "
            "문맥과 상황에 맞게 적절한 어휘와 표현을 선택하세요. "
            "사회적 맥락과 관계를 고려하여 적절한 존대 표현을 사용하세요."
        ),

        "n1": (
            "고급 한국어 표현과 관용적 어휘를 활용하여 자연스럽게 번역하세요. "
            "원문의 뉘앙스와 문체를 최대한 살리면서 세련된 한국어로 표현하세요. "
            "문화적 맥락을 고려하여 한국어 독자에게 자연스럽게 전달되도록 번역하세요. "
            "상황과 인간관계에 적합한 존대 표현과 말투를 정확히 구사하세요."
        ),

        "standard": (
            "자연스러운 한국어 표현으로 의미를 정확히 전달하세요. "
            "문맥에 맞게 적절한 어휘와 표현을 선택하세요. "
            "한국어 화자가 실제로 사용하는 자연스러운 표현을 사용하세요. "
            "상황에 맞는 적절한 존대 표현과 말투를 사용하세요."
        ),
    }


class LLMService:
    """Service for interacting with the Gemini API."""

    @staticmethod
    def initialize_gemini():
        """Initialize Gemini API."""
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
    def call_llm(
            prompt: str,
            temperature: float = Config.DEFAULT_TEMPERATURE
    ) -> Optional[str]:
        """
        Call the Gemini API and get a response with improved response handling.
        """
        if not Config.GEMINI_API_KEY:
            app.logger.error("API key not configured. Set GEMINI_API_KEY environment variable.")
            return None

        # Initialize Gemini API if not already initialized
        LLMService.initialize_gemini()

        retry_count = 0

        while retry_count < Config.MAX_RETRIES:
            try:
                # Load the model
                model = genai.GenerativeModel(Config.MODEL_NAME)

                # Generate content
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048
                    }
                )

                if response:
                    # 개선된 응답 텍스트 추출 로직
                    try:
                        # 먼저 response.text 시도
                        return response.text
                    except Exception as text_error:
                        app.logger.warning(f"response.text failed: {text_error}")

                        # 대안 방법: candidates를 통한 접근
                        try:
                            if response.candidates and len(response.candidates) > 0:
                                candidate = response.candidates[0]
                                if candidate.content and candidate.content.parts:
                                    # parts에서 텍스트 추출
                                    text_parts = []
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text_parts.append(part.text)

                                    if text_parts:
                                        return ''.join(text_parts)
                                    else:
                                        app.logger.error("No text found in response parts")
                                else:
                                    app.logger.error("No content or parts in candidate")
                            else:
                                app.logger.error("No candidates in response")
                        except Exception as parts_error:
                            app.logger.error(f"Failed to extract text from parts: {parts_error}")

                    # 모든 방법이 실패한 경우
                    app.logger.error("All text extraction methods failed")

                else:
                    app.logger.error("Empty response from Gemini API")

                retry_count += 1
                if retry_count < Config.MAX_RETRIES:
                    time.sleep(2)

            except Exception as e:
                app.logger.error(f"Gemini API call error: {str(e)}")
                retry_count += 1
                if retry_count < Config.MAX_RETRIES:
                    time.sleep(2)
                else:
                    return None

        return None


class JapaneseExampleGenerator:
    """Generator for Japanese example sentences with Korean translations."""

    def __init__(self):
        pass

    @staticmethod
    def generate_examples(
            word: str,
            difficulty: str = Config.DEFAULT_DIFFICULTY,
            num_examples: int = Config.DEFAULT_NUM_EXAMPLES,
            max_retries: int = 3  # 재시도 횟수 증가
    ) -> List[Dict[str, str]]:
        """
        Generate natural Japanese example sentences with guaranteed count.
        """
        # 더 많은 예문을 요청하여 필터링 후에도 충분히 남도록 함
        requested_num = min(num_examples + 3, 8)  # 3개 더 요청 (최대 8개)

        # 재시도 로직
        for attempt in range(max_retries + 1):
            try:
                # Get level-specific components
                level_text = Config.LEVEL_DESCRIPTIONS.get(difficulty, Config.LEVEL_DESCRIPTIONS["standard"])
                instruction_detail = Config.DETAILED_INSTRUCTIONS.get(difficulty,
                                                                      Config.DETAILED_INSTRUCTIONS["standard"])
                variation_instruction = Config.USAGE_VARIATIONS.get(difficulty, Config.USAGE_VARIATIONS["standard"])
                korean_translation_guide = Config.KOREAN_TRANSLATION_GUIDELINES.get(difficulty,
                                                                                    Config.KOREAN_TRANSLATION_GUIDELINES[
                                                                                        "standard"])

                # 재시도일 경우 온도 값을 약간 변경하여 다양한 결과 유도
                temperature = Config.DEFAULT_TEMPERATURE
                if attempt > 0:
                    temperature = min(0.9, Config.DEFAULT_TEMPERATURE + 0.1 * attempt)
                    app.logger.info(f"Retry attempt {attempt} with temperature {temperature}")

                # Build the prompt with requested_num (더 많은 예문 요청)
                prompt = JapaneseExampleGenerator._build_example_prompt(
                    word, level_text, instruction_detail, variation_instruction, korean_translation_guide, requested_num
                )

                # Call the language model and parse its response
                response = LLMService.call_llm(prompt, temperature=temperature)

                if not response:
                    app.logger.warning(f"No response from LLM (attempt {attempt + 1}/{max_retries + 1})")
                    continue

                # Parse examples from the response
                examples = JapaneseExampleGenerator._parse_examples(response, word)

                # 유효한 예시가 충분한지 확인
                valid_examples = [ex for ex in examples if
                                  ex["japanese"] != "例文の生成に失敗しました。" and
                                  ex["japanese"] != "適切な例文の生成に失敗しました。"]

                app.logger.info(f"Generated {len(valid_examples)} valid examples out of {num_examples} requested")

                # 목표 개수에 도달했는지 확인
                if len(valid_examples) >= num_examples:
                    # 정확히 요청된 개수만 반환
                    return valid_examples[:num_examples]

                # 부족한 경우 추가 생성 시도
                if len(valid_examples) > 0 and attempt < max_retries:
                    remaining = num_examples - len(valid_examples)
                    app.logger.info(f"Need {remaining} more examples, attempting additional generation")

                    # 추가 예문 생성
                    additional_prompt = JapaneseExampleGenerator._build_example_prompt(
                        word, level_text, instruction_detail, variation_instruction,
                        korean_translation_guide, remaining + 2  # 여유분 추가
                    )

                    additional_temp = min(0.95, temperature + 0.15)
                    additional_response = LLMService.call_llm(additional_prompt, temperature=additional_temp)

                    if additional_response:
                        additional_examples = JapaneseExampleGenerator._parse_examples(additional_response, word)
                        additional_valid = [ex for ex in additional_examples if
                                            ex["japanese"] != "例文の生成に失敗しました。" and
                                            ex["japanese"] != "適切な例文の生成に失敗しました。"]

                        # 기존 예문과 합치기
                        valid_examples.extend(additional_valid)
                        app.logger.info(f"Added {len(additional_valid)} more examples, total: {len(valid_examples)}")

                        # 목표 개수에 도달했으면 반환
                        if len(valid_examples) >= num_examples:
                            return valid_examples[:num_examples]

                # 여전히 부족한 경우 다음 시도로
                if len(valid_examples) < max(1, int(num_examples * 0.5)):
                    continue

                # 최소한의 예문이라도 있으면 반환 (부분적 성공)
                if valid_examples:
                    app.logger.warning(f"Returning {len(valid_examples)} examples instead of {num_examples}")
                    return valid_examples

            except Exception as e:
                app.logger.error(f"Error during example generation (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                if attempt < max_retries:
                    continue

        # 모든 시도가 실패한 경우
        app.logger.error("All generation attempts failed")
        return [
            {
                "context": "",
                "japanese": "例文の生成に失敗しました。",
                "korean": "예문 생성에 실패했습니다。"
            }
        ]

    @staticmethod
    def _build_example_prompt(
            word: str,
            level_text: str,
            instruction_detail: str,
            variation_instruction: str,
            korean_translation_guide: str,
            num_examples: int
    ) -> str:
        """
        Build the detailed prompt for the LLM to generate examples.

        Args:
            word: Target Japanese word
            level_text: Description of the JLPT level
            instruction_detail: Detailed grammar instructions
            variation_instruction: Usage variation hints
            korean_translation_guide: Guidelines for Korean translation
            num_examples: Number of examples to generate

        Returns:
            Formatted prompt string
        """
        return f"""
        # Japanese Example Sentence Generator

        ## Role
        You are an experienced Japanese language teacher creating authentic example sentences for learners.

        ## Target Word
        "{word}"

        ## Requirements
        - Generate EXACTLY {num_examples} example sentences using the word "{word}"
        - Use {level_text} Japanese appropriate for the learner's level
        - {instruction_detail}
        - {variation_instruction}
        - EVERY sentence must be COMPLETE. Never leave sentences unfinished or ending with "...".
        - Ensure each sentence has a proper subject, predicate, and any necessary objects or complements.

        ## Context Guidelines
        - Create realistic situations from daily Japanese life (home, work, school, restaurants, etc.)
        - Show the word being used in different contexts and by different speakers
        - If the word has multiple meanings, demonstrate each major usage
        - EXTREMELY IMPORTANT: Ensure each example is semantically correct and natural in Japanese
        - For verbs like "食べる" (to eat), only use objects that can actually be eaten in Japanese culture
        - For action verbs, ensure the actions are logical and natural in real-life contexts

        ## Korean Translation Guidelines
        - {korean_translation_guide}
        - Focus on natural Korean translation that a native Korean speaker would actually use
        - Don't translate word-for-word, but convey the meaning in natural Korean
        - Maintain the appropriate level of formality from the Japanese sentence
        - Use Korean expressions and idioms that match the context where appropriate
        - Make sure Korean translations are complete sentences that fully express the meaning

        ## Output Format
        For each example, strictly follow this exact format with no deviation:

        1. Context: [Brief context in English - 1-2 sentences maximum]
        Japanese: [Complete natural Japanese sentence that uses the word "{word}"]
        Korean: [Complete natural Korean translation]

        2. Context: [Brief context in English - 1-2 sentences maximum]
        Japanese: [Complete natural Japanese sentence that uses the word "{word}"]
        Korean: [Complete natural Korean translation]

        ## Semantic Guidelines for Nouns
        - For nouns such as "図書館" (library), "料理" (cooking), "友達" (friend):
        - Place them in realistic contexts:
            - Location nouns (e.g. 図書館, 公園) should appear with verbs like "に行く" (to go to) or "で勉強する" (to study at)
            - Animate nouns (e.g. 友達, 犬) should be used as subjects (が) or objects (を) in varied sentence roles
            - Abstract nouns (e.g. 自由, 幸福) should pair with conceptual verbs like "を感じる" (to feel) or "を求める" (to seek)
        - Demonstrate correct particle usage for nouns: が, を, に, の, で, etc.
        - Vary modifiers:
        - Adjective modifiers (e.g. "大きな家", "美味しい料理")
        - Quantifier expressions (e.g. "三冊の本", "二人の友達")
        - If a noun has multiple senses, present distinct situations illustrating each sense

        ## Verb and Tense Guidelines
        - 動詞「食べる」「飲む」などは、具体的な食べ物・飲み物を目的語として使用してください (例: 「リンゴを食べる」、「コーヒーを飲む」).
        - 抽象的・イベント名詞（「ピクニック」など）は目的語にしないでください.
        - 文脈に応じて時制を明確に区別してください。完了した動作には過去形(～た), 習慣・未来には現在形(～る/～ます形)を使用してください。
        - 行動動詞を使うときは、必ず具体的な対象や場所を一緒に明示してください (例: 「公園でサンドイッチを食べる」、「図書館で勉強する」).
        - 全ての文は完全な文として終了させてください。「...」で終わる不完全な文は使わないでください。

        ## Important Notes
        - ALWAYS include the word "{word}" in each Japanese sentence
        - Each example MUST include: Context, Japanese sentence, Korean translation
        - Number each example consecutively (1-{num_examples})
        - Focus on natural, authentic Japanese as actually used by native speakers
        - DO NOT include any romanization, pronunciation guide, or English translation in the Japanese sentence
        - DO NOT put romaji, hiragana readings, or any non-Japanese explanations in parentheses
        - Japanese sentences should contain ONLY Japanese characters, nothing else
        - Do not add explanations or notes between examples
        - DO NOT add any numbering or markers at the end of Korean translations
        - ALL sentences must be COMPLETE with proper ending. NO ellipses (...) or unfinished sentences.
        """.strip()

    # 추가 검증 함수
    @staticmethod
    def _validate_semantics(examples: List[Dict[str, str]], word: str) -> List[Dict[str, str]]:
        """
        Validates the semantic naturalness of Japanese sentences.

        This function checks for semantically inappropriate verb-object combinations,
        unnatural patterns, and formatting issues in both Japanese and Korean text.

        Args:
            examples: List of example dictionaries generated by LLM
            word: Target Japanese word to validate

        Returns:
            List of validated examples that pass semantic and formatting checks
        """

        class SemanticValidator:
            """Helper class for semantic validation of Japanese sentences"""

            # Semantic constraints for specific verbs and their inappropriate objects
            VERB_OBJECT_CONSTRAINTS = {
                "食べる": {  # "to eat"
                    "invalid_objects": [
                        "日本語", "勉強", "宿題", "問題", "試験", "テスト", "文法", "言語",
                        "車", "電車", "家", "ビル", "学校", "会社", "音楽", "映画", "テレビ"
                    ],
                    "pattern_template": r"{obj}を食べる"
                },
                "飲む": {  # "to drink"
                    "invalid_objects": [
                        "電車", "自転車", "車", "本", "映画", "テレビ", "家", "ビル",
                        "宿題", "問題", "音楽", "ゲーム"
                    ],
                    "pattern_template": r"{obj}を飲む"
                },
                "避ける": {  # "to avoid"
                    "invalid_patterns": [r"部屋を避ける"]  # "avoid room" is unnatural
                }
            }

            # General unnatural patterns regardless of specific words
            UNNATURAL_PATTERNS = [
                r'部屋を避ける',  # "avoid room"
                r'車を食べる',  # "eat car"
                r'家を飲む',  # "drink house"
                r'問題を歩く',  # "walk problem"
            ]

            # Korean translation quality patterns to filter out
            KOREAN_FORMATTING_ISSUES = [
                r'\d+\.\s*\*+',  # Numbered markers with asterisks
                r'^\d+\.',  # Lines starting with numbers
                r'[ぁ-んァ-ン一-龥]',  # Japanese characters in Korean text
            ]

            @classmethod
            def has_semantic_violation(cls, japanese_text: str, target_word: str) -> bool:
                """Check if the sentence has semantic violations for the target word"""
                if target_word not in cls.VERB_OBJECT_CONSTRAINTS:
                    return False

                constraints = cls.VERB_OBJECT_CONSTRAINTS[target_word]

                # Check verb-object constraints
                if "invalid_objects" in constraints:
                    pattern_template = constraints["pattern_template"]
                    for invalid_obj in constraints["invalid_objects"]:
                        pattern = pattern_template.format(obj=re.escape(invalid_obj))
                        if re.search(pattern, japanese_text):
                            return True

                # Check predefined invalid patterns
                if "invalid_patterns" in constraints:
                    for pattern in constraints["invalid_patterns"]:
                        if re.search(pattern, japanese_text):
                            return True

                return False

            @classmethod
            def has_unnatural_patterns(cls, japanese_text: str) -> bool:
                """Check for general unnatural patterns in Japanese text"""
                return any(re.search(pattern, japanese_text) for pattern in cls.UNNATURAL_PATTERNS)

            @classmethod
            def has_korean_formatting_issues(cls, korean_text: str) -> bool:
                """Check for formatting issues in Korean translation"""
                return any(re.search(pattern, korean_text) for pattern in cls.KOREAN_FORMATTING_ISSUES)

            @classmethod
            def clean_korean_text(cls, korean_text: str) -> str:
                """Clean and normalize Korean text"""
                # Remove multiple consecutive newlines
                cleaned = re.sub(r'\n{2,}', '\n', korean_text)
                # Remove any remaining Japanese characters
                cleaned = re.sub(r'[ぁ-んァ-ン一-龥]', '', cleaned)
                return cleaned.strip()

        def is_valid_example(example: Dict[str, str]) -> bool:
            """
            Validate a single example for basic requirements and semantic correctness

            Args:
                example: Dictionary containing 'japanese' and 'korean' keys

            Returns:
                bool: True if example passes all validation checks
            """
            japanese_text = example.get('japanese', '').strip()
            korean_text = example.get('korean', '').strip()

            # Basic validation: both texts must exist and have minimum length
            if not japanese_text or not korean_text:
                return False

            if len(japanese_text) < 5 or len(korean_text) < 5:
                return False

            # Target word must be present in Japanese sentence
            if word not in japanese_text:
                return False

            # Check for semantic violations
            if SemanticValidator.has_semantic_violation(japanese_text, word):
                return False

            # Check for unnatural patterns
            if SemanticValidator.has_unnatural_patterns(japanese_text):
                return False

            # Check Korean formatting issues
            if SemanticValidator.has_korean_formatting_issues(korean_text):
                return False

            return True

        # Process and validate examples
        validated_examples = []

        for example in examples:
            if is_valid_example(example):
                # Clean Korean text before adding to validated list
                example['korean'] = SemanticValidator.clean_korean_text(example['korean'])
                validated_examples.append(example)

        # Return validated examples or error message if none are valid
        if validated_examples:
            return validated_examples
        else:
            return [{
                "context": "",
                "japanese": "適切な例文の生成に失敗しました。",
                "korean": "적절한 예문 생성에 실패했습니다."
            }]

    @staticmethod
    def _parse_examples(response_text: str, word: str) -> List[Dict[str, str]]:
        """
        Extract Japanese examples and translations from the LLM response.

        Args:
            response_text: Text response from the LLM
            word: Target Japanese word for validation

        Returns:
            List of dictionaries with context, Japanese, and Korean fields
        """
        examples = []

        # Debug - log response for troubleshooting
        app.logger.debug("Response from LLM (first 500 chars): " +
                         (response_text[:500] + "..." if len(response_text) > 500 else response_text))

        # Main pattern: Find examples with Context/Japanese/Korean labels
        main_pattern = r'(?:\d+\.\s*)?Context:\s*(.*?)\s*Japanese:\s*(.*?)\s*Korean:\s*(.*?)(?=\s*(?:\d+\.\s*)?Context:|\s*$)'
        matches = re.findall(main_pattern, response_text, re.DOTALL)

        if matches:
            app.logger.debug(f"Found {len(matches)} examples with the main pattern")
            for match in matches:
                context, japanese, korean = match

                # 불완전한 문장 확인 (1): 문장이 "..." 또는 "…"로 끝나는 경우
                if japanese.strip().endswith('...') or japanese.strip().endswith('…'):
                    app.logger.debug(f"Skipping incomplete sentence: {japanese}")
                    continue

                # 불완전한 문장 확인 (2): 문장 길이가 너무 짧은 경우
                if len(japanese.strip()) < 10:
                    app.logger.debug(f"Skipping too short sentence: {japanese}")
                    continue

                # 로마자 표기 제거 (괄호와 괄호 안의 내용 제거)
                japanese = re.sub(r'\s*\([^)]*\)', '', japanese)

                # Clean Korean translation from any Japanese characters
                korean = re.sub(r'[ぁ-んァ-ン一-龥]', '', korean)

                # 추가 정제: 번역에서 숫자와 마커 제거
                korean = re.sub(r'\d+\.\s*\*+', '', korean)
                korean = re.sub(r'^\d+\.\s*', '', korean)

                # 여러 줄 바꿈 정리
                korean = re.sub(r'\n{2,}', '\n', korean)

                examples.append({
                    "context": context.strip(),
                    "japanese": japanese.strip(),
                    "korean": korean.strip()
                })

        # If main pattern doesn't find anything, try alternative pattern
        if not examples:
            # Numbered example pattern
            numbered_pattern = r'(\d+)\.[\s\n]+(.*?)[\s\n]+Japanese:[\s\n]+(.*?)[\s\n]+Korean:[\s\n]+(.*?)(?=[\s\n]+\d+\.|\s*$)'
            matches = re.findall(numbered_pattern, response_text, re.DOTALL)

            if matches:
                app.logger.debug(f"Found {len(matches)} examples with the numbered pattern")
                for match in matches:
                    number, context, japanese, korean = match

                    # 불완전한 문장 확인 (1): 문장이 "..." 또는 "…"로 끝나는 경우
                    if japanese.strip().endswith('...') or japanese.strip().endswith('…'):
                        app.logger.debug(f"Skipping incomplete sentence: {japanese}")
                        continue

                    # 불완전한 문장 확인 (2): 문장 길이가 너무 짧은 경우
                    if len(japanese.strip()) < 10:
                        app.logger.debug(f"Skipping too short sentence: {japanese}")
                        continue

                    # 로마자 표기 제거 (괄호와 괄호 안의 내용 제거)
                    japanese = re.sub(r'\s*\([^)]*\)', '', japanese)

                    # Clean Korean translation from any Japanese characters
                    korean = re.sub(r'[ぁ-んァ-ン一-龥]', '', korean)

                    # 추가 정제: 번역에서 숫자와 마커 제거
                    korean = re.sub(r'\d+\.\s*\*+', '', korean)
                    korean = re.sub(r'^\d+\.\s*', '', korean)

                    # 여러 줄 바꿈 정리
                    korean = re.sub(r'\n{2,}', '\n', korean)

                    examples.append({
                        "context": context.strip(),
                        "japanese": japanese.strip(),
                        "korean": korean.strip()
                    })

        # 다시 한번 불완전한 문장 필터링
        filtered_examples = []
        for example in examples:
            japanese = example.get('japanese', '')
            korean = example.get('korean', '')

            # 문장이 끝나지 않은 경우 필터링
            if (japanese.endswith('...') or japanese.endswith('…') or
                    len(japanese) < 10 or word not in japanese or
                    not korean or len(korean) < 5):
                continue

            filtered_examples.append(example)

        # Semantic validation using _validate_semantics
        if filtered_examples:
            validated_examples = JapaneseExampleGenerator._validate_semantics(filtered_examples, word)

            # Check if we have enough examples after validation
            if validated_examples and not all(
                    ex.get('japanese') == "適切な例文の生成に失敗しました。" for ex in validated_examples):
                return validated_examples

        # 최소 예시 개수가 충족되지 않은 경우, 다시 한번 LLM 호출 시도
        if len(filtered_examples) < 2:
            app.logger.warning("Not enough valid examples. Returning error message.")
            return [{"context": "", "japanese": "例文の生成に失敗しました。", "korean": "예문 생성에 실패했습니다."}]

        return filtered_examples

    @staticmethod
    def _build_word_info_prompt(word: str) -> str:
        """
        Build the prompt for getting word information.

        Args:
            word: Japanese word to analyze

        Returns:
            Formatted prompt string
        """
        return f"""
        # Comprehensive Japanese Word Analysis Request

        ## Target Word
        {word}

        ## Task
        Provide comprehensive and accurate linguistic information about this Japanese word.

        ## Content Requirements
        Please include the following information:

        ### 1. Basic Information
        - Part of speech: Precise classification (noun, verb, adjective, adverb, etc.)
        - Reading: How the word is pronounced
        - Kanji representation: If applicable
        - Etymology: Origins of the word if available
        - Usage frequency: How commonly the word is used in daily conversation, literature, business, etc.
        - Politeness level: Honorific, neutral, casual, etc.

        ### 2. Meaning Information
        - Basic meaning: Core definition
        - Additional meanings: Secondary definitions
        - Nuances and connotations: Subtle implications
        - Contextual variations: How meaning changes in different contexts

        ### 3. Usage Examples
        - Daily conversation examples (2-3)
        - Written language examples (1-2)
        - Special situation examples (1-2)
        - Idiomatic expressions or proverbs using the word (1-2 if applicable)

        ### 4. Related Expressions
        - Synonyms (3-4 words)
        - Antonyms (1-2 if applicable)
        - Related expressions and collocations (3-4)
        - Derivative words (2-3 if applicable)

        ### 5. Grammatical Information

        #### For Verbs:
        - Conjugation group (Godan, Ichidan, Irregular, etc.)
        - Main conjugation forms:
          * Dictionary form
          * Masu form
          * Te form
          * Nai form (negative)
          * Ta form (past)
          * Volitional form
          * Potential form
          * Passive form
          * Causative form
          * Conditional form
          * Imperative form
        - Transitivity (transitive/intransitive distinction)
        - Common particle patterns and grammatical constructions

        #### For Adjectives:
        - Adjective type (i-adjective/na-adjective)
        - Main conjugation forms:
          * Basic form
          * Polite form (desu form)
          * Negative form
          * Past form
          * Negative past form
          * Adverbial form
          * Conditional form
        - Common combination patterns

        #### For Nouns:
        - Particle combination patterns
        - Common verbs used with this noun
        - Compound word formation
        - Count/non-count distinction (if applicable)

        ### 6. Cultural/Contextual Information
        - Cultural background
        - Historical development
        - Generational/regional usage differences
        - Social implications

        ## Output Format
        Structure your response in two sections - first in Japanese, then the same information in Korean:

        === 日本語 ===
        【品詞】
        【読み方】
        【漢字表記】(該当する場合)
        【語源】(該当する場合)
        【基本的な意味】
        【追加的な意味・ニュアンス】
        【使用頻度・場面】
        【敬語レベル】
        【日常会話での例文】
        【文語体の例文】
        【特殊な状況での例文】
        【慣用句・ことわざ】(該当する場合)
        【類義語・同義語】
        【対義語】(該当する場合)
        【関連表現・コロケーション】
        【派生語】(該当する場合)
        【文法情報】
        * 活用グループ(動詞の場合)
        * 主な活用形
        * 自動詞・他動詞の区別(動詞の場合)
        * 結合パターン
        【文化的・社会的情報】

        === 한국어 ===
        【품사】
        【읽는 법】
        【한자 표기】(해당되는 경우)
        【어원】(해당되는 경우)
        【기본적인 의미】
        【추가적인 의미・뉘앙스】
        【사용 빈도・상황】
        【경어 수준】
        【일상 대화 예문】
        【문어체 예문】
        【특수 상황 예문】
        【관용구・속담】(해당되는 경우)
        【유의어・동의어】
        【반의어】(해당되는 경우)
        【관련 표현・연어】
        【파생어】(해당되는 경우)
        【문법 정보】
        * 활용 그룹(동사인 경우)
        * 주요 활용형
        * 자동사・타동사 구별(동사인 경우)
        * 결합 패턴
        【문화적・사회적 정보】

        ## Important Notes
        - DO NOT include any English in your actual response content (only in the template headers)
        - Keep your entire response under 600 words
        - Focus on clear, accurate linguistic information
        - Use natural, everyday Japanese for examples
        - You don't need to fill in all categories if they're not applicable to this word
        - Omit irrelevant information based on the word type (e.g., skip verb conjugation info for nouns)
        """.strip()

    @staticmethod
    def format_output(examples: List[Dict[str, str]]) -> str:
        """
        Format examples into a concise output string.

        Args:
            examples: List of example dictionaries
            include_furigana: Whether to add furigana readings (optional feature for later)

        Returns:
            Formatted output string
        """
        result = ""
        for i, example in enumerate(examples, 1):
            result += f"📝 예시 {i}\n"

            if example.get('japanese') and example['japanese'].strip():
                result += f"🇯🇵 일본어: {example['japanese']}\n"
                result += f"🇰🇷 한국어: {example['korean']}\n\n"

        # If no examples were formatted
        if not result:
            result = "예문을 생성하지 못했습니다.\n"

        return result


# Flask routes

@app.route('/api/examples', methods=['POST'])
def api_examples():
    """API endpoint for generating examples."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON 데이터가 필요합니다."}), 400

        word = data.get('word')
        level = data.get('level', Config.DEFAULT_DIFFICULTY)
        format_type = data.get('format', 'simple')  # 'simple', 'with_hiragana', 'with_context'

        if not word:
            return jsonify({"error": "단어가 필요합니다."}), 400

        if level.lower() not in Config.VALID_LEVELS:
            level = Config.DEFAULT_DIFFICULTY

        # Generate examples
        examples = JapaneseExampleGenerator.generate_examples(word, level.lower())

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
            # 현재는 단순히 일본어와 한국어만 포함
            examples_clean = [
                {
                    "japanese_example": ex.get("japanese", "").strip(),
                    "korean_translation": ex.get("korean", "").strip()
                }
                for ex in examples
            ]
            # 향후 히라가나 변환 API 등을 통해 독음 추가 가능
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


# Command-line interface
def main():
    """
    Command-line interface for the Japanese example generator.
    Handles user input, word lookup, example generation, etc.
    """
    print("\n" + "=" * 50)
    print("🇯🇵 일본어 예문 생성기 🇯🇵")
    print("자연스러운 일본어 예문과 한국어 번역을 생성합니다")
    print("=" * 50)
    print("• 단어 정보를 보려면 'info [단어]'를 입력하세요")
    print("• 종료하려면 Ctrl+C를 누르세요")
    print("=" * 50 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("\n👉 일본어 단어를 입력하세요 (또는 'info [단어]'): ").strip()

            # Skip empty input
            if not user_input:
                continue

            # Check if user wants word information
            if user_input.lower().startswith('info '):
                word = user_input[5:].strip()  # Extract word after 'info '
                if not word:
                    print("❌ 'info' 뒤에 단어를 입력해주세요")
                    continue

                print(f"\n📚 '{word}'에 대한 정보를 검색 중...")
                word_info = JapaneseExampleGenerator.get_word_info(word)
                print("\n" + word_info)
                continue

            # Use entered word for example generation
            word = user_input

            # Get JLPT level input
            difficulty_input = input("🎯 JLPT 레벨 (n5/n4/n3/n2/n1) 또는 'standard' (기본값: n3): ").strip().lower()

            # Validate and set difficulty
            if difficulty_input and difficulty_input in Config.VALID_LEVELS:
                difficulty = difficulty_input
            else:
                if difficulty_input and difficulty_input not in Config.VALID_LEVELS:
                    print(f"⚠️ 유효하지 않은 레벨 '{difficulty_input}', 기본값(n3)을 사용합니다.")
                difficulty = Config.DEFAULT_DIFFICULTY

            # Display format
            display_difficulty = difficulty.upper() if difficulty in ["n5", "n4", "n3", "n2", "n1"] else difficulty
            print(f"\n🔍 '{word}'를 사용한 {display_difficulty} 레벨 예문을 생성 중...")

            # Generate and display examples
            try:
                examples = JapaneseExampleGenerator.generate_examples(word, difficulty)
                print("\n✨ 생성된 예문:")
                print(JapaneseExampleGenerator.format_output(examples))

            except Exception as e:
                print(f"❌ 예문 생성 오류: {str(e)}")
                print("다른 단어나 설정으로 다시 시도해주세요.")

        except KeyboardInterrupt:
            print("\n\n👋 일본어 예문 생성기를 이용해주셔서 감사합니다! 안녕히 가세요.")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            print("다시 시도해주세요.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
