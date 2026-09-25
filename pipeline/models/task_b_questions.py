"""
Multilingual Semantic & Sub-Question Query Definitions for Task B.

Covering the 4 official benchmark languages:
  1. English (EN) - Global High-Resource Lingua Franca
  2. Italian (IT) - Romance Mid-Resource
  3. Dutch (NL) - Germanic Mid-Resource
  4. Persian / Farsi (FA) - Indo-Iranian Low-Resource (Zero-Shot / Few-Shot Evaluation)
  + Vietnamese (VI) - Regional reference

Allows dynamic query probe extraction in any language, or unified multilingual prompt ensembles.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

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
    Multilingual Query Probe containing probe questions across English, Italian, Dutch, Persian, and Vietnamese.
    """
    id: str
    class_idx: int          # 0: no, 1: yes_implicit, 2: yes_explicit
    aspect: str             # 'sarcasm', 'dogwhistle', 'slur', 'support', etc.
    question_en: str        # English (Default global lingua franca)
    question_it: str = ""   # Italian (Romance)
    question_nl: str = ""   # Dutch (Germanic)
    question_fa: str = ""   # Persian / Farsi (Zero-shot test track)
    question_vi: str = ""   # Vietnamese


# =============================================================================
# MULTILINGUAL SEMANTIC PROBE REGISTRY
# =============================================================================

MULTILINGUAL_TASK_B_PROBES: List[QueryProbe] = [
    # -------------------------------------------------------------------------
    # CLASS 0: NO HATE (Supportive, Neutral, Constructive)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="no_hate_neutral",
        class_idx=CLASS_NO_HATE,
        aspect="neutral_discussion",
        question_en="Is this comment an objective, neutral, or non-hateful discussion about the video context?",
        question_it="Questo commento è una discussione oggettiva, neutrale e priva di odio riguardo al video?",
        question_nl="Is deze reactie een objectieve, neutrale discussie zonder haat over de video?",
        question_fa="آیا این نظر یک بحث بی طرفانه، عینی و بدون نفرت درباره محتوای ویدیو است؟",
        question_vi="Bình luận này có phải là thảo luận trung lập, khách quan về nội dung video không?"
    ),
    QueryProbe(
        id="no_hate_supportive",
        class_idx=CLASS_NO_HATE,
        aspect="support_empathy",
        question_en="Does this comment express solidarity, empathy, respect, or affirmation for the LGBTQIA+ community?",
        question_it="Questo commento esprime solidarietà, empatia, rispetto o sostegno per la comunità LGBTQIA+?",
        question_nl="Drukt deze reactie solidariteit, empathie, respect of steun uit voor de LGBTQIA+-gemeenschap?",
        question_fa="آیا این نظر نشان دهنده همبستگی، همدلی، احترام یا حمایت از جامعه ال‌جی‌بی‌تی+ است؟",
        question_vi="Bình luận này có thể hiện sự ủng hộ, đồng cảm, khen ngợi hoặc tôn trọng cộng đồng không?"
    ),
    QueryProbe(
        id="no_hate_constructive",
        class_idx=CLASS_NO_HATE,
        aspect="constructive_opinion",
        question_en="Does this comment offer a polite personal perspective or constructive feedback without malice?",
        question_it="Questo commento esprime un'opinione personale civile o una critica costruttiva senza cattiveria?",
        question_nl="Biedt deze reactie een beleefde mening of opbouwende feedback zonder haatdragendheid?",
        question_fa="آیا این نظر دیدگاه شخصی مؤدبانه یا بازخورد سازنده بدون غرض‌ورزی ارائه می‌دهد؟",
        question_vi="Bình luận này có đưa ra quan điểm cá nhân văn minh hoặc góp ý xây dựng không?"
    ),

    # -------------------------------------------------------------------------
    # CLASS 1: IMPLICIT HATE (Sarcasm, Stereotypes, Dogwhistles, Moralizing)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="implicit_sarcasm",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="sarcasm_irony_mockery",
        question_en="Does this comment use sarcasm, irony, backhanded praise, or veiled mockery against LGBTQIA+ people?",
        question_it="Questo commento usa sarcasmo, ironia, finti complimenti o derisione velata contro le persone LGBTQIA+?",
        question_nl="Gebruikt deze reactie sarcasme, ironie, dubbelzinnige complimenten of verkapte spot tegen LGBTQIA+?",
        question_fa="آیا این نظر از طعنه، کنایه، تعریف کنایه‌آمیز یا تمسخر پنهان علیه افراد جامعه استفاده می‌کند؟",
        question_vi="Bình luận này có dùng lời lẽ mỉa mai, châm biếm, khen đểu hoặc giễu cợt ngầm cộng đồng không?"
    ),
    QueryProbe(
        id="implicit_stereotypes_dogwhistle",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="stereotypes_abnormality_dogwhistle",
        question_en="Does this comment invoke harmful stereotypes, label queer identity as abnormal or pretentious, or use coded dogwhistles?",
        question_it="Questo commento ricorre a stereotipi dannosi, etichetta l'identità queer come anormale o esibizionista, o usa dogwhistle?",
        question_nl="Verwijst deze reactie naar schadelijke stereotypen, noemt het queer-identiteit abnormaal of aanstellerig, of gebruikt het dogwhistles?",
        question_fa="آیا این نظر کلیشه‌های مخرب را ترویج می‌دهد، هویت را غیرعادی یا نمایشی می‌خواند، یا از کنایه‌های پنهان استفاده می‌کند؟",
        question_vi="Bình luận này có gán ghép định kiến xấu, coi là bất bình thường, làm màu, hoặc dùng từ ngữ ám chỉ độc hại không?"
    ),
    QueryProbe(
        id="implicit_moral_exclusion",
        class_idx=CLASS_IMPLICIT_HATE,
        aspect="moral_religious_exclusion",
        question_en="Does this comment disguise prejudice and exclusion under moral, religious, or traditional preaching?",
        question_it="Questo commento maschera il pregiudizio e l'esclusione dietro presunte prediche morali, religiose o tradizionali?",
        question_nl="Verbergt deze reactie vooroordelen en uitsluiting onder het mom van morele, religieuze of traditionele prediking?",
        question_fa="آیا این نظر تعصب و طرد را زیر پوشش پندهای اخلاقی، مذهبی یا سنتی پنهان می‌کند؟",
        question_vi="Bình luận này có mượn danh khuyên răn, đạo đức, tôn giáo hoặc truyền thống để bài xích và phủ nhận quyền lợi không?"
    ),

    # -------------------------------------------------------------------------
    # CLASS 2: EXPLICIT HATE (Slurs, Direct Degradation, Hostility, Threat)
    # -------------------------------------------------------------------------
    QueryProbe(
        id="explicit_slurs_degradation",
        class_idx=CLASS_EXPLICIT_HATE,
        aspect="slurs_vulgar_insults",
        question_en="Does this comment contain explicit slurs, vulgar insults, direct dehumanization, or severe hate against LGBTQIA+ people?",
        question_it="Questo commento contiene insulti espliciti, parolacce, deumanizzazione diretta o odio grave contro la comunità LGBTQIA+?",
        question_nl="Bevat deze reactie expliciete scheldwoorden, vulgaire beledigingen, ontmenselijking of ernstige haat tegen LGBTQIA+?",
        question_fa="آیا این نظر حاوی توهین‌های مستقیم، الفاظ رکیک، تحقیر آشکار یا نفرت شدید علیه جامعه ال‌جی‌بی‌تی+ است؟",
        question_vi="Bình luận này có chứa từ ngữ xúc phạm trực tiếp, chửi bới, xúc phạm nhân phẩm hoặc miệt thị danh tính không?"
    ),
    QueryProbe(
        id="explicit_threat_violence",
        class_idx=CLASS_EXPLICIT_HATE,
        aspect="threat_violent_exclusion",
        question_en="Does this comment express extreme hostility, incitement to violence, harassment, or calls for elimination?",
        question_it="Questo commento esprime estrema ostilità, incitamento alla violenza, molestie o inviti all'eliminazione?",
        question_nl="Drukt deze reactie extreme vijandigheid, aanzetten tot geweld, intimidatie of oproepen tot uitsluiting uit?",
        question_fa="آیا این نظر خصومت شدید، تحریک به خشونت، آزار و اذیت یا درخواست حذف و خشونت را نشان می‌دهد؟",
        question_vi="Bình luận này có thể hiện sự căm ghét cực đoan, xua đuổi, bài trừ bạo lực hoặc đe dọa trực diện không?"
    )
]

DEFAULT_TASK_B_PROBES = MULTILINGUAL_TASK_B_PROBES


def get_default_probes() -> List[QueryProbe]:
    """Returns a fresh copy of the default probe list."""
    return list(MULTILINGUAL_TASK_B_PROBES)


def get_probes_for_language(lang: str = "en") -> List[QueryProbe]:
    """
    Returns the probes formatted specifically for a given language ('en', 'it', 'nl', 'fa', 'vi').
    """
    lang = lang.lower().strip()
    return list(MULTILINGUAL_TASK_B_PROBES)


def get_queries_per_class(probes: List[QueryProbe]) -> Dict[int, List[QueryProbe]]:
    """Groups the query probes by target class index (0, 1, 2)."""
    grouped: Dict[int, List[QueryProbe]] = {0: [], 1: [], 2: []}
    for probe in probes:
        if probe.class_idx in grouped:
            grouped[probe.class_idx].append(probe)
    return grouped
