import os
import sys
from flask import Flask, request, jsonify
from typing import Dict, List

# 필수 모듈 가져오기 및 검증
HomonymExampleGenerator = None
HomonymConfig = None
JapaneseExampleGenerator = None
ExampleConfig = None

# 첫 번째 파일에서 HomonymExampleGenerator 가져오기
try:
    from homonym_processor import HomonymExampleGenerator, Config as HomonymConfig
except ImportError as e:
    print(f"❌ homonym_processor.py 파일을 찾을 수 없습니다: {e}")
    print("📝 첫 번째 파일을 homonym_processor.py로 저장하고 Flask 라우트 부분을 제거해주세요")
    HomonymExampleGenerator = None
    HomonymConfig = None

# 두 번째 파일에서 JapaneseExampleGenerator 가져오기
try:
    from example_generator import JapaneseExampleGenerator, Config as ExampleConfig
except ImportError as e:
    print(f"❌ example_generator.py 파일을 찾을 수 없습니다: {e}")
    print("📝 두 번째 파일을 example_generator.py로 저장하고 Flask 라우트 부분을 제거해주세요")
    JapaneseExampleGenerator = None
    ExampleConfig = None

# 필수 모듈 검증
missing_modules = []
if HomonymExampleGenerator is None:
    missing_modules.append("homonym_processor.py")
if JapaneseExampleGenerator is None:
    missing_modules.append("example_generator.py")

if missing_modules:
    print(f"\n💥 필수 모듈이 누락되었습니다: {', '.join(missing_modules)}")
    print("서비스를 시작할 수 없습니다. 누락된 파일들을 준비한 후 다시 실행해주세요.")
    sys.exit(1)

app = Flask(__name__)


class MainConfig:
    """메인 애플리케이션 설정"""
    DEFAULT_PORT = 3000
    DEFAULT_HOST = '0.0.0.0'
    DEBUG_MODE = False

    # 지원되는 JLPT 레벨
    VALID_LEVELS = ["n5", "n4", "n3", "n2", "n1", "standard"]
    DEFAULT_LEVEL = "n3"

    # 응답 형식 옵션
    VALID_FORMATS = ["simple", "with_context", "with_hiragana"]
    DEFAULT_FORMAT = "simple"


def format_examples_by_type(examples: List[Dict], format_type: str) -> List[Dict]:
    """
    응답 형식에 따라 예문 데이터 포맷팅

    Args:
        examples: 원본 예문 데이터
        format_type: 응답 형식 (simple, with_context, with_hiragana)

    Returns:
        포맷팅된 예문 리스트
    """
    try:
        if format_type == 'simple':
            # 가장 기본 형식: 일본어와 한국어만
            return [
                {
                    "japanese_example": ex.get("japanese", "").strip(),
                    "korean_translation": ex.get("korean", "").strip()
                }
                for ex in examples
                if ex.get("japanese") and ex.get("korean")
            ]

        elif format_type == 'with_context':
            # 컨텍스트 포함 형식
            return [
                {
                    "context": ex.get("context", "").strip(),
                    "japanese_example": ex.get("japanese", "").strip(),
                    "korean_translation": ex.get("korean", "").strip()
                }
                for ex in examples
                if ex.get("japanese") and ex.get("korean")
            ]

        elif format_type == 'with_hiragana':
            # 히라가나 독음 포함 형식 (향후 구현 예정)
            return [
                {
                    "japanese_example": ex.get("japanese", "").strip(),
                    "korean_translation": ex.get("korean", "").strip(),
                    "hiragana_reading": "히라가나 변환 기능 준비 중"  # 향후 구현
                }
                for ex in examples
                if ex.get("japanese") and ex.get("korean")
            ]

        else:
            # 기본값: simple 형식
            return format_examples_by_type(examples, 'simple')

    except Exception as e:
        app.logger.error(f"예문 포맷팅 중 오류: {str(e)}")
        return []


@app.route('/api/homonym', methods=['POST'])
def api_homonym():
    """
    동음이의어 전용 API 엔드포인트 (첫 번째 모듈)

    요청 형식:
    {
        "word": "일본어 단어",
        "level": "n5|n4|n3|n2|n1|standard" (선택사항, 기본값: n3),
        "format": "simple|with_context|with_hiragana" (선택사항, 기본값: simple)
    }
    """
    try:
        # 모듈 로드 확인
        if HomonymExampleGenerator is None:
            return jsonify({
                "success": False,
                "error": "동음이의어 처리 모듈이 로드되지 않았습니다. homonym_processor.py 파일을 확인해주세요."
            }), 503

        # 요청 데이터 검증
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "JSON 데이터가 필요합니다.",
                "message": "요청 본문에 유효한 JSON 데이터를 포함해주세요."
            }), 400

        word = data.get('word')
        if not word:
            return jsonify({
                "error": "단어가 필요합니다.",
                "message": "'word' 필드에 일본어 단어를 입력해주세요."
            }), 400

        # 파라미터 검증 및 기본값 설정
        level = data.get('level', MainConfig.DEFAULT_LEVEL).lower()
        if level not in MainConfig.VALID_LEVELS:
            level = MainConfig.DEFAULT_LEVEL

        format_type = data.get('format', MainConfig.DEFAULT_FORMAT).lower()
        if format_type not in MainConfig.VALID_FORMATS:
            format_type = MainConfig.DEFAULT_FORMAT

        app.logger.info(f"동음이의어 요청: word={word}, level={level}, format={format_type}")

        # HomonymExampleGenerator를 사용하여 동음이의어 분석
        examples = HomonymExampleGenerator.generate_homonym_examples(word, level.lower())

        return jsonify(examples)

    except Exception as e:
        app.logger.error(f"동음이의어 API 오류: {str(e)}")
        return jsonify({
            "error": "동음이의어 분석 중 오류가 발생했습니다.",
            "message": f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    일반 예문 생성 전용 API 엔드포인트 (두 번째 모듈)

    요청 형식:
    {
        "word": "일본어 단어",
        "level": "n5|n4|n3|n2|n1|standard" (선택사항, 기본값: n3),
        "format": "simple|with_context|with_hiragana" (선택사항, 기본값: simple)
    }
    """
    try:
        # 모듈 로드 확인
        if JapaneseExampleGenerator is None:
            return jsonify({
                "success": False,
                "error": "예문 생성 모듈이 로드되지 않았습니다. example_generator.py 파일을 확인해주세요."
            }), 503

        # 요청 데이터 검증
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "JSON 데이터가 필요합니다.",
                "message": "요청 본문에 유효한 JSON 데이터를 포함해주세요."
            }), 400

        word = data.get('word')
        if not word:
            return jsonify({
                "error": "단어가 필요합니다.",
                "message": "'word' 필드에 일본어 단어를 입력해주세요."
            }), 400

        # 파라미터 검증 및 기본값 설정
        level = data.get('level', MainConfig.DEFAULT_LEVEL).lower()
        if level not in MainConfig.VALID_LEVELS:
            level = MainConfig.DEFAULT_LEVEL

        format_type = data.get('format', MainConfig.DEFAULT_FORMAT).lower()
        if format_type not in MainConfig.VALID_FORMATS:
            format_type = MainConfig.DEFAULT_FORMAT

        app.logger.info(f"예문 생성 요청: word={word}, level={level}, format={format_type}")

        # JapaneseExampleGenerator를 사용하여 예문 생성
        examples = JapaneseExampleGenerator.generate_examples(
            word=word,
            difficulty=level,
            num_examples=5,  # 기본 5개 예문 생성
            max_retries=2
        )

        # 응답 형식에 따라 데이터 구성
        formatted_examples = format_examples_by_type(examples, format_type)

        # Return examples
        return jsonify({"examples": formatted_examples})

    except Exception as e:
        app.logger.error(f"예문 생성 API 오류: {str(e)}")
        return jsonify({
            "error": "예문 생성 중 오류가 발생했습니다.",
            "message": f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        }), 500


def main():
    """메인 실행 함수"""
    # 환경 변수에서 설정 읽기
    port = int(os.environ.get("PORT", MainConfig.DEFAULT_PORT))
    host = os.environ.get("HOST", MainConfig.DEFAULT_HOST)
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print("\n" + "=" * 70)
    print("🇯🇵 일본어 예문 생성 통합 서비스 🇯🇵")
    print("=" * 70)
    print(f"🚀 서버 시작: http://{host}:{port}")
    print("=" * 70)
    print("🔧 API 엔드포인트:")
    print("  • 전용 모드:")
    print(f"    - POST http://{host}:{port}/api/homonym (동음이의어)")
    print(f"    - POST http://{host}:{port}/api/generate (일반 예문)")
    print("=" * 70)
    print("📋 기능:")
    print("  • 동음이의어 분석 및 구별 예문 생성")
    print("  • 일반 단어 예문 생성")
    print("=" * 70)
    print("💡 사용 예시:")
    print("  # 동음이의어 모드")
    print(f"  curl -X POST http://{host}:{port}/api/homonym \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"word": "きく", "level": "n3"}\'')
    print()
    print("  # 일반 예문 모드")
    print(f"  curl -X POST http://{host}:{port}/api/generate \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"word": "食べる", "level": "n3"}\'')
    print("=" * 70 + "\n")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 서비스를 종료합니다. 안녕히 가세요!")
    except Exception as e:
        print(f"\n❌ 서버 시작 중 오류: {e}")


if __name__ == '__main__':
    main()