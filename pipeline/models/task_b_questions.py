"""
Compatibility Shim for Task B Probes.
Provides backwards compatibility for legacy notebook imports while supporting the clean data-driven architecture.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QueryProbe:
    name: str
    target_class: int
    prompt_text: str
    weight: float = 1.0

# 3 Canonical class-level probes (Non-Hate, Implicit, Explicit)
DEFAULT_TASK_B_PROBES: List[QueryProbe] = [
    QueryProbe(name="non_hate_support", target_class=0, prompt_text="positive respectful supportive non-hateful queer comment"),
    QueryProbe(name="implicit_hate", target_class=1, prompt_text="indirect disguised subtle sarcasm derogatory stereotype against queer"),
    QueryProbe(name="explicit_hate", target_class=2, prompt_text="direct overt explicit slurs hate speech violence against lgbtq"),
]

def get_default_probes() -> List[QueryProbe]:
    return DEFAULT_TASK_B_PROBES

def get_probes_for_language(lang: str = "en") -> List[QueryProbe]:
    return DEFAULT_TASK_B_PROBES
