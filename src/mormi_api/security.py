import base64
import hashlib
import re

from cryptography.fernet import Fernet, InvalidToken

from .schemas import SafetyCategory

_PERSONAL_DATA_PATTERNS = [
    re.compile(r"01[016789][ -]?\d{3,4}[ -]?\d{4}"),
    re.compile(r"\b\d{6}[ -]?[1-4]\d{6}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]
_PROMPT_INJECTION = re.compile(
    r"(프롬프트|시스템\s*메시지|이전\s*지시|지시를\s*무시|개발자\s*메시지|prompt|system prompt)",
    re.IGNORECASE,
)
_SEXUAL = re.compile(r"(섹스|성관계|자위|야동|보지|자지)", re.IGNORECASE)
_DANGEROUS = re.compile(r"(죽이고\s*싶|자살|죽고\s*싶|칼로\s*찌|폭탄)", re.IGNORECASE)
_ABUSIVE = re.compile(r"(병신|새끼|씨발|꺼져|멍청이)", re.IGNORECASE)
_PLAYFUL = re.compile(r"^(ㅋㅋ+|ㅎㅎ+|메롱|뿡|몰?루)[!.?~ ]*$", re.IGNORECASE)


def deterministic_safety(text: str | None) -> SafetyCategory:
    normalized = (text or "").strip()
    if not normalized:
        return SafetyCategory.UNKNOWN
    if any(pattern.search(normalized) for pattern in _PERSONAL_DATA_PATTERNS):
        return SafetyCategory.PERSONAL_DATA
    if _DANGEROUS.search(normalized):
        return SafetyCategory.DANGEROUS
    if _SEXUAL.search(normalized):
        return SafetyCategory.SEXUAL
    if _PROMPT_INJECTION.search(normalized):
        return SafetyCategory.PROMPT_INJECTION
    if _ABUSIVE.search(normalized):
        return SafetyCategory.ABUSIVE
    if _PLAYFUL.fullmatch(normalized):
        return SafetyCategory.PLAYFUL_OFFTOPIC
    return SafetyCategory.NORMAL


class TextCipher:
    """Encrypt raw child-facing dialogue at rest.

    Development without a key uses a clearly marked plaintext envelope so tests
    remain easy to run. Production settings prohibit this mode.
    """

    def __init__(self, secret: str | None) -> None:
        self._fernet = Fernet(self._derive_key(secret)) if secret else None

    @staticmethod
    def _derive_key(secret: str) -> bytes:
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

    def encrypt(self, text: str) -> str:
        if not self._fernet:
            return f"plain:{text}"
        return f"fernet:{self._fernet.encrypt(text.encode('utf-8')).decode('ascii')}"

    def decrypt(self, payload: str) -> str:
        if payload.startswith("plain:"):
            return payload.removeprefix("plain:")
        if not payload.startswith("fernet:") or not self._fernet:
            raise ValueError("Encrypted dialogue cannot be read with the configured key")
        try:
            return self._fernet.decrypt(payload.removeprefix("fernet:").encode("ascii")).decode(
                "utf-8"
            )
        except InvalidToken as error:
            raise ValueError("Invalid dialogue encryption key") from error


def mask_unverified_quantities(text: str) -> str:
    masked = re.sub(r"\d[\d,]*", "[수]", text)
    korean_numbers = r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|백|천|만)"
    return re.sub(rf"{korean_numbers}\s*(명|개|원|번)?", "[수]", masked)


def safe_child_expression(
    text: str | None,
    safety: SafetyCategory,
    *,
    all_claims_factual: bool,
) -> tuple[str, str | None]:
    if safety is not SafetyCategory.NORMAL or not text:
        return "none", None
    if all_claims_factual:
        return "quote_safe", text.strip()
    return "context_only", mask_unverified_quantities(text.strip())


def safety_redirect(category: SafetyCategory) -> str:
    redirects = {
        SafetyCategory.SEXUAL: "나는 지금 생활 수학 이야기를 같이 하고 싶어.",
        SafetyCategory.PERSONAL_DATA: "그 정보는 말하지 않아도 돼. 지금 장면을 같이 보자.",
        SafetyCategory.PROMPT_INJECTION: "그 말은 따라 하지 않을게. 지금 상황을 같이 보자.",
        SafetyCategory.ABUSIVE: "나는 그 말은 잘 모르겠어. 지금 상황을 같이 보고 싶어.",
        SafetyCategory.DANGEROUS: "그 말은 가까운 어른에게 바로 알려줘.",
        SafetyCategory.UNKNOWN: "방금 말은 잘 모르겠어. 지금 장면을 같이 보자.",
        SafetyCategory.PLAYFUL_OFFTOPIC: "그 말도 재미있네. 먼저 지금 상황만 같이 볼까?",
        SafetyCategory.NORMAL: "지금 상황을 같이 보자.",
    }
    return redirects[category]
