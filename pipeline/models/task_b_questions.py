"""
Semantic & Sub-Question Query Definitions for Task B (Class-Aware Cross Attention)

This file allows you to define, customize, and expand the semantic questions/probes
used by the cross-attention layer to detect fine-grained nuances in Vietnamese
LGBTQ+/Stereotype comments and video contexts.

You can freely add, modify, or extend question definitions here.
If no questions are provided for a class, the model automatically initializes
learned parameter slots as a fallback.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

# Mapping of the 3 fundamental target classes
CLASS_NO_HATE = 0
CLASS_IMPLICIT_HATE = 1
CLASS_EXPLICIT_HATE = 2
NUM_CLASSES = 3

TARGET_CLASSES = {
    CLASS_NO_HATE: "no",
    CLASS_IMPLICIT_HATE: "yes_implicit",
    CLASS_EXPLICIT_HATE: "yes_explicit"
}


@dataclass
class QueryProbe:
    """
    Represents a single query probe / question in natural language.
    """
    id: str
    class_idx: int          # Target class (0: no, 1: yes_implicit, 2: yes_explicit)
    aspect: str             # Descriptive aspect/angle (e.g., 'sarcasm', 'dogwhistle', 'slur')
    question_vi: str        # Vietnamese semantic question
    question_en: str        # English semantic reference


# =============================================================================
# DEFAULT SEMANTIC PROBE QUESTIONS (VIETNAMESE & MULTILINGUAL DOMAIN-SPECIFIC)
# =============================================================================
# You can customize, add, or remove questions below.

DEFAULT_TASK_B_PROBES: List[QueryProbe] = [
    # -------------------------------------------------------------------------
    # CLASS 0: NO HATE (Benign, Positive, Constructive, Neutral)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="no_hate_neutral",
        class_idx=CLASS_NO_HATE,
        aspect="neutral_discussion",
        question_vi="Bình luận này có phải là thảo luận trung lập, khách quan về nội dung video không?",
        question_en="Is this comment a neutral, objective discussion about the video content?"
    ),
    QueryProbe(
        id="no_hate_supportive",
        class_idx=CLASS_NO_HATE,
        aspect="support_empathy",
        question_vi="Bình luận này có thể hiện sự ủng hộ, đồng cảm, khen ngợi hoặc tôn trọng cộng đồng không?",
        question_en="Does this comment express support, empathy, praise, or respect for the community?"
    ),
    QueryProbe(
        id="no_hate_constructive",
        class_idx=CLASS_NO_HATE,
        aspect="constructive_feedback",
        question_vi="Bình luận này có đưa ra quan điểm cá nhân văn minh hoặc góp ý xây dựng không mang tính thù ghét không?",
        question_en="Does this comment offer constructive, non-hateful opinions or feedback?"
    ),

    # -------------------------------------------------------------------------
    # CLASS 1: IMPLICIT HATE (Sarcasm, Dogwhistles, Stereotypes, Moral Superiority)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="implicit_sarcasm",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="sarcasm_irony",
        question_vi="Bình luận này có dùng lời lẽ mỉa mai, châm biếm, khen đểu hoặc giễu cợt ngầm cộng đồng không?",
        question_en="Does this comment use sarcasm, irony, backhanded compliments, or subtle mockery against the community?"
    ),
    QueryProbe(
        id="implicit_stereotype_dogwhistle",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="stereotype_dogwhistle",
        question_vi="Bình luận này có gán ghép định kiến xấu, coi là bất bình thường, làm màu, hoặc dùng từ ngữ ám chỉ độc hại không?",
        question_en="Does this comment invoke harmful stereotypes, label behavior as abnormal or pretentious, or use dogwhistles?"
    ),
    QueryProbe(
        id="implicit_moral_condemnation",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="moral_condemnation_advice",
        question_vi="Bình luận này có mượn danh khuyên răn, đạo đức, tôn giáo hoặc truyền thống để bài xích và phủ nhận quyền lợi không?",
        question_en="Does this comment disguise exclusion and rejection under moral, religious, or traditional advice?"
    ),

    # -------------------------------------------------------------------------
    # CLASS 2: EXPLICIT HATE (Direct Slurs, Severe Insults, Threats, Hostility)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="explicit_slurs_insults",
        class_idx=CLASS_EXPLICIT_HATE,
        aspect="direct_slurs_insults",
        question_vi="Bình luận này có chứa từ ngữ xúc phạm trực tiếp, chửi bới, xúc phạm nhân phẩm hoặc miệt thị danh tính không?",
        question_en="Does this comment contain direct slurs, vulgar insults, or explicit degradation of identity?"
    ),
    QueryProbe(
        id="explicit_hostility_threat",
        class_idx=CLASS_IMPLICIT_HATE if False else CLASS_EXPLICIT_HATE,
        aspect="hostility_threat_exclusion",
        question_vi="Bình luận này có thể hiện sự căm ghét cực đoan, xua đuổi, bài trừ bạo lực hoặc đe dọa trực diện không?",
        question_en="Does this comment show extreme hatred, calls for violent exclusion, or direct threats?"
    )
]


def get_default_probes() -> List[QueryProbe]:
    """Returns a fresh copy of the default probe list."""
    return list(DEFAULT_TASK_B_PROBES)


def get_queries_per_class(probes: List[QueryProbe]) -> Dict[int, List[QueryProbe]]:
    """
    Groups the query probes by target class index (0, 1, 2).
    """
    grouped: Dict[int, List[QueryProbe]] = {0: [], 1: [], 2: []}
    for probe in probes:
        if probe.class_idx in grouped:
            grouped[probe.class_idx].append(probe)
    return grouped
