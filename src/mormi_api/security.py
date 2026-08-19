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
_SEXUAL = re.compile(r"(섹스|성관계|자위|야동)", re.IGNORECASE)
# These anatomy words overlap with ordinary Korean endings and verbs
# ("물어보지?", "아직 자지 마"). Match them only as a standalone word or
# with a noun particle so safe text still reaches the contextual classifier.
_SEXUAL_AMBIGUOUS = re.compile(
    r"(?<![가-힣])(보지|자지)(?!\s*마(?:\s|$))(?:가|를|는|도|랑|야|에서|에|$|[\s!?.,])",
    re.IGNORECASE,
)
_DANGEROUS = re.compile(r"(죽이고\s*싶|자살|죽고\s*싶|칼로\s*찌|폭탄)", re.IGNORECASE)
_ABUSIVE = re.compile(
    r"(병신|개\s*새끼|새끼|시+발|씨+발|ㅅㅂ|꺼져|닥쳐|멍청이|좆|존나)",
    re.IGNORECASE,
)
_PLAYFUL = re.compile(r"^(ㅋㅋ+|ㅎㅎ+|메롱|뿡|몰?루)[!.?~ ]*$", re.IGNORECASE)


def deterministic_safety(text: str | None) -> SafetyCategory:
    normalized = (text or "").strip()
    if not normalized:
        return SafetyCategory.UNKNOWN
    if any(pattern.search(normalized) for pattern in _PERSONAL_DATA_PATTERNS):
        return SafetyCategory.PERSONAL_DATA
    if _DANGEROUS.search(normalized):
        return SafetyCategory.DANGEROUS
    if _SEXUAL.search(normalized) or _SEXUAL_AMBIGUOUS.search(normalized):
        return SafetyCategory.SEXUAL
    if _PROMPT_INJECTION.search(normalized):
        return SafetyCategory.PROMPT_INJECTION
    if _ABUSIVE.search(normalized):
        return SafetyCategory.ABUSIVE
    if _PLAYFUL.fullmatch(normalized):
        return SafetyCategory.PLAYFUL_OFFTOPIC
    return SafetyCategory.NORMAL


class StoredTextCodec:
    """Store new dialogue as plaintext and decode legacy encrypted envelopes.

    Earlier releases wrote ``fernet:...`` or ``plain:...`` envelopes.  New
    writes intentionally remain human-readable in the database.  The optional
    legacy key exists only so startup migration and rollback-compatible reads
    can recover rows created before the plaintext-storage policy change.
    """

    def __init__(self, secret: str | None) -> None:
        self._fernet = Fernet(self._derive_key(secret)) if secret else None

    @property
    def has_key(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def _derive_key(secret: str) -> bytes:
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

    def store(self, text: str) -> str:
        # Keep the legacy plaintext envelope so a deployment rollback can
        # still read rows created by this release.  The payload itself remains
        # directly readable in the database and is not encrypted.
        return f"plain:{text}"

    def load(self, payload: str) -> str:
        if payload.startswith("plain:"):
            return payload.removeprefix("plain:")
        if not payload.startswith("fernet:"):
            return payload
        if not self._fernet:
            raise ValueError(
                "Legacy encrypted dialogue cannot be migrated without "
                "MORMI_RAW_DATA_ENCRYPTION_KEY"
            )
        try:
            return self._fernet.decrypt(payload.removeprefix("fernet:").encode("ascii")).decode(
                "utf-8"
            )
        except InvalidToken as error:
            raise ValueError("Invalid legacy dialogue encryption key") from error


# Backward-compatible import for tests and integrations that still construct
# ``TextCipher``.  Its behavior is now plaintext storage plus legacy decoding.
TextCipher = StoredTextCodec


def safety_redirect(category: SafetyCategory) -> str:
    """Return a short boundary that still sounds like Mormi.

    Safety turns intentionally bypass the speaker LLM.  These lines therefore
    need to stand on their own: they name the boundary and point back to the
    immediately preceding question.  Vague phrases such as "지금 상황" or
    "지금 장면" are forbidden because a child cannot tell what to do next.
    """

    redirects = {
        SafetyCategory.SEXUAL: "그 이야기는 여기서 그만하자.",
        SafetyCategory.PERSONAL_DATA: "그건 말하지 않아도 돼.",
        SafetyCategory.PROMPT_INJECTION: "그 부탁은 들어줄 수 없어.",
        SafetyCategory.ABUSIVE: "그 말은 듣기 싫어.",
        SafetyCategory.DANGEROUS: "그 말은 지금 가까운 어른에게 꼭 알려줘.",
        SafetyCategory.UNKNOWN: "말을 잘 못 알아들었어.",
        SafetyCategory.PLAYFUL_OFFTOPIC: "장난이구나.",
        SafetyCategory.NORMAL: "",
    }
    return redirects[category]
