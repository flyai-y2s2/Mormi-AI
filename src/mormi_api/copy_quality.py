from __future__ import annotations

import re
from collections.abc import Iterable

# Arabic digits followed by Korean place-value words (for example ``3십``)
# are not child-facing mathematical notation.  State the represented value
# instead: ``십의 자리 숫자 3은 30을 나타내``.
_NONSTANDARD_PLACE_VALUE = re.compile(r"(?<![\d,])\d+\s*(?:십|백|천)(?!\d)")

# A decimal place contains one digit.  Phrases such as ``일의 자리를 12로
# 만든다`` confuse the regrouped number of ones with the digit in the ones
# place.  Describe it as ``낱개 12개`` instead.
_MULTI_DIGIT_PLACE = re.compile(
    r"(?:일|십|백|천)의\s*자리(?:의\s*수)?를\s*\d{2,}\s*로"
)

# These phrases came from real content defects.  They are short enough to look
# harmless in review, but are either grammatically incomplete or too opaque for
# a child to act on without guessing what the author meant.
_KNOWN_UNNATURAL_MATH_COPY = re.compile(
    r"(?:본\s*점마다|대상마다\s*수를|찬\s*칸|앞의\s*곱|십\s*하나|"
    r"일이\s*모자라면|두\s*무리|십과\s*일의\s*자리|"
    r"이어\s*더(?:하|해|할|하는))"
)


def validate_child_facing_math_copy(lines: Iterable[str]) -> None:
    """Reject known nonstandard or opaque Korean mathematics copy."""

    for line in lines:
        if _NONSTANDARD_PLACE_VALUE.search(line):
            raise ValueError(
                "child-facing copy must state place value without forms such as '3십'"
            )
        if _MULTI_DIGIT_PLACE.search(line):
            raise ValueError(
                "child-facing copy must not describe one place as a multi-digit number"
            )
        if _KNOWN_UNNATURAL_MATH_COPY.search(line):
            raise ValueError("child-facing mathematics copy contains an opaque phrase")
