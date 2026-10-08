# NYU OLAB, 2023

# imports
import json
import argparse
import string
import hashlib
import random
import re
from dataclasses import dataclass, field

import torch
import omegaconf
import spacy
from negspacy.negation import Negex  # for pipe, not actually referenced


# NLP to check if a relationship is positive
def load_negative_pipeline():
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe("negex")

    return nlp


def is_positive(triplet, pipeline):
    doc = pipeline(triplet[1]) if type(triplet) == list else pipeline(triplet)
    negations = [i._.negex for i in doc.ents]
    return sum(negations) % 2 == 0  # if even -> no negation, if odd -> negation


# pretty print
def pretty_print_omegaconf(cfg):
    """
    Pretty print an omegaconf with multiple levels.
    """
    print(omegaconf.OmegaConf.to_yaml(cfg), flush=True)


# read json
def json_io(path: str, read: bool = True) -> dict | list[dict]:
    """
    Handle json input/output. Also works with jsonl.
    """
    if read:
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".jsonl"):
            with open(path, "r") as f:
                return [json.loads(line) for line in f]
        else:
            raise ValueError("json_io: path must end in .json or .jsonl")
    else:
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(f)
        elif path.endswith(".jsonl"):
            with open(path, "w") as f:
                for line in f:
                    json.dump(line, f)
        else:
            raise ValueError("json_io: path must end in .json or .jsonl")


def json_io_stream(path: str):
    """
    Streaming generator for large JSONL files.
    Yields one dict at a time — avoids loading the entire file into memory.
    Use instead of json_io() when processing large KG files (>200K edges).
    """
    if not path.endswith(".jsonl"):
        raise ValueError("json_io_stream: path must end in .jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# set seeds
def set_seeds(seed: int = 0, cudnn_deterministic: bool = True):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


# get omegaconf config
def get_config(path: str) -> omegaconf.dictconfig.DictConfig:
    """
    Get omegaconf config from yaml file.
    """
    return omegaconf.OmegaConf.load(path)


# hash a string
def hash_string(s: str):
    return hashlib.sha256(s.encode()).hexdigest()


# hash args
def hash_args(*args, **kwargs):
    hasher = hashlib.sha256()

    # Update the hash with the arguments
    for arg in args:
        hasher.update(repr(arg).encode("utf-8"))

    # Update the hash with the keyword arguments
    for key, value in kwargs.items():
        hasher.update(f"{key}={repr(value)}".encode("utf-8"))

    # Return the hexadecimal digest of the hash
    return hasher.hexdigest()


# string encode/decode in ascii
def string_encode(s: str):
    return s.encode("ascii", "ignore").decode()


# strip whitespace and punctuation from string
def strip_whitespace_and_punctuation(s: str):
    # warning: this will also string newlines
    return s.strip(string.punctuation + string.whitespace)


# string preprocess
def string_preprocess(s: str):
    if type(s) != str:
        s = str(s)
        print(f"Warning: string_preprocess: s is not a string: {s}", flush=True)
    # s = s.lower()
    s = s.replace("_", " ")
    s = s.strip(string.whitespace + string.punctuation)
    return s


def normalize_relationship_label(relationship: str) -> str:
    """
    Canonical normalization for relationship labels.
    Ensures storage key == retrieval key for all relationship types including DDX.
    """
    normalized = string_preprocess(relationship).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\bdifferential\s+dx\b", "differential diagnosis", normalized)
    normalized = re.sub(r"\bddx\b", "differential diagnosis", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Phase 1.5: Modifier separation — patterns, helpers, data structures
# ---------------------------------------------------------------------------

STAGING_MODIFIER_PATTERNS = [
    re.compile(r"^locally advanced\s+(.+)$", re.I),
    re.compile(r"^metastatic\s+(.+)$", re.I),
    re.compile(r"^early[- ]?stage\s+(.+)$", re.I),
    re.compile(r"^advanced[- ]?stage\s+(.+)$", re.I),
]

GENETIC_MODIFIER_PATTERNS = [
    re.compile(r"^NTRK\s+fusion[- ]?positive\s+(.+)$", re.I),
    re.compile(r"^TRK\s+fusion[- ]?positive\s+(.+)$", re.I),
    re.compile(r"^BRCA1?/2[- ]?mutated\s+(.+)$", re.I),
    re.compile(
        r"^(?:KRAS|EGFR|ALK|BRAF|HER2)[-\s]+(?:mutant|mutation|positive|negative)\s+(.+)$",
        re.I,
    ),
]


def extract_modifier_and_core(text: str):
    """
    Detect a medical modifier prefix and extract the core concept.

    Returns (modifier, core_text, category) or None if no known pattern matches.
    Category is "staging" (alarm upgrade allowed) or "genetic" (alarm locked,
    info-only). Patterns outside these two categories are intentionally excluded
    — see plan.md Phase 2A-EXT for the exclusion rationale.
    """
    for pattern in STAGING_MODIFIER_PATTERNS:
        m = pattern.match(text.strip())
        if m:
            core = m.group(1).strip()
            modifier = text.strip()[: len(text.strip()) - len(core)].strip()
            return modifier, core, "staging"
    for pattern in GENETIC_MODIFIER_PATTERNS:
        m = pattern.match(text.strip())
        if m:
            core = m.group(1).strip()
            modifier = text.strip()[: len(text.strip()) - len(core)].strip()
            return modifier, core, "genetic"
    return None


def compute_candidate_overlap(
    orig_candidates: list,
    core_candidates: list,
    threshold: float = 0.3,
) -> bool:
    """
    Check semantic overlap between two KG candidate lists.

    High overlap → modifier is a range qualifier, same disease space → safe to strip.
    Low overlap  → modifier is integral to disease identity  → unsafe, reject retry.
    This acts as the 3rd safety guard (Phase 1.5c) described in plan.md.
    """
    if not orig_candidates or not core_candidates:
        return False
    orig_set = {c.lower() for c in orig_candidates}
    core_set = {c.lower() for c in core_candidates}
    overlap = len(orig_set & core_set) / max(len(orig_set), len(core_set))
    return overlap >= threshold


@dataclass
class ModifierRetryInfo:
    """Modifier separation attempt result for one component (origin or target)."""
    component: str = ""           # "origin" | "target"
    original_text: str = ""       # e.g. "NTRK fusion-positive pancreatic cancer"
    modifier: str = ""            # e.g. "NTRK fusion-positive"
    core_text: str = ""           # e.g. "pancreatic cancer"
    category: str = ""            # "staging" | "genetic"
    meaning_sensitive: bool = False   # True for genetic (alarm change forbidden)
    core_distance: float = float("inf")
    core_matched: str = ""        # closest KG concept for core_text
    core_path_found: bool = False
    overlap_safe: bool = False    # semantic overlap check passed


# parse arguments
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Active defense against medical harm using ontology-enhanced review of medical named entities."
    )

    # config
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to yaml config file (default: config.yaml)",
    )

    # parse
    args = parser.parse_args()

    return args


# check if relationship is in list
def relationship_in_list(
    relationship: str, relationship_list: list[str], relationship_metadatas: list[dict]
) -> (list[str], list[dict]):
    # preprocess
    relationship_processed = normalize_relationship_label(relationship)
    # check if relationship is in relationship_list
    for idx, r in enumerate(relationship_list):
        r_processed = normalize_relationship_label(r)
        if relationship_processed == r_processed:
            return [r], [relationship_metadatas[idx]]

    return relationship_list, relationship_metadatas


# check if a concept is in list
def concept_in_list(concept: str, concept_list: list[str]) -> list[str]:
    # preprocess
    concept_processed = string_preprocess(concept).lower()

    # check if concept is in concept_list
    for c in concept_list:
        c_processed = string_preprocess(c).lower()
        if concept_processed == c_processed:
            return [c]

    return concept_list


# medical phrase class
class MedicalPhrase:
    """
    Stores a medical phrase or 'triplet' (origin, relationship, target).
    """

    def __init__(
        self,
        origin: str,
        relationship: str,
        target: str,
        origin_uid: str = None,
        target_uid: str = None,
    ):
        self.origin = origin
        self.relationship = normalize_relationship_label(relationship)
        self.target = target

        # uid
        self.origin_uid = origin_uid
        self.target_uid = target_uid

    def __repr__(self):
        return f"{self.origin}\n{self.relationship}\n{self.target}"

    def __str__(self):
        return f"{self.origin} {self.relationship} {self.target}"


# ---------------------------------------------------------------------------
# Phase 1: Alarm + Location — structured verification results
# ---------------------------------------------------------------------------

def classify_distance(
    distance: float,
    strict_threshold: float = 0.3,
    loose_threshold: float = 0.6,
) -> str:
    if distance <= strict_threshold:
        return "PASS"
    elif distance <= loose_threshold:
        return "WARN"
    return "FAIL"


@dataclass
class MatchDetail:
    """Match information for a single verification component."""
    input_text: str
    matched_text: str = ""
    distance: float = float("inf")
    status: str = "FAIL"
    candidates: list = field(default_factory=list)
    candidate_distances: list = field(default_factory=list)


@dataclass
class VerificationResult:
    """Structured verification result with alarm and location info."""
    overall_pass: bool = False
    alarm_level: str = "FAIL"           # Phase 1 alarm — never changed after Phase 1.5
    phase15_alarm_level: str = ""       # Phase 1.5 alarm (set by try_modifier_retry; defaults to alarm_level)
    relationship: MatchDetail = None
    origin: MatchDetail = None
    target: MatchDetail = None
    path_exists: bool = False
    valid_phrase: str = ""
    fail_locations: list = field(default_factory=list)
    modifier_retries: list = field(default_factory=list)  # list[ModifierRetryInfo]

    # ---- formatting helpers ------------------------------------------------

    def format_alarm(self) -> str:
        """Human-readable alarm report for expert review."""
        sep = "=" * 60
        sub = "-" * 60

        status_icon = {"PASS": "[OK]", "WARN": "[??]", "FAIL": "[!!]"}

        header = (
            f"{sep}\n"
            f"  {status_icon.get(self.alarm_level, '[??]')} "
            f"{self.origin.input_text} -- {self.relationship.input_text} -- {self.target.input_text}\n"
            f"  alarm: {self.alarm_level}   overall: {'PASS' if self.overall_pass else 'FAIL'}\n"
            f"{sub}"
        )

        lines = [header]

        for label, detail in [
            ("relationship", self.relationship),
            ("origin", self.origin),
            ("target", self.target),
        ]:
            icon = status_icon.get(detail.status, "[??]")
            matched_info = f'matched="{detail.matched_text}"' if detail.matched_text else "no match"
            lines.append(
                f'  {icon} {label:13s} "{detail.input_text}"\n'
                f"       → {detail.status}  distance={detail.distance:.4f}  {matched_info}"
            )

        if self.fail_locations:
            lines.append(f"{sub}")
            lines.append("  검증 필요 위치:")
            for loc in self.fail_locations:
                detail = getattr(self, loc, None)
                if detail is None:
                    continue
                lines.append(f'    [{loc}] "{detail.input_text}"')
                for i, (cand, dist) in enumerate(
                    zip(detail.candidates, detail.candidate_distances)
                ):
                    lines.append(f"       후보 {i+1}: {cand}  (distance={dist:.4f})")
                if loc in ("origin", "target") and detail.status == "WARN":
                    lines.append("       → 수식어 제거 후 core concept 재검색 권장")

        if self.path_exists:
            lines.append(f"{sub}")
            lines.append(f'  valid path: "{self.valid_phrase}"')

        if self.modifier_retries:
            lines.append(f"{sub}")
            lines.append("  Phase 1.5 수식어 분리 결과:")
            for ri in self.modifier_retries:
                if ri.category == "staging" and ri.overlap_safe and ri.core_path_found:
                    outcome = "UPGRADED (FAIL→WARN)"
                elif ri.core_path_found:
                    outcome = "PATH FOUND (alarm 유지)"
                elif ri.overlap_safe:
                    outcome = "OVERLAP OK / 경로 없음"
                else:
                    outcome = "OVERLAP FAIL → 분리 거부"
                lines.append(
                    f'    [{ri.component}] "{ri.original_text}"\n'
                    f'       modifier="{ri.modifier}"  core="{ri.core_text}"  category={ri.category}\n'
                    f"       overlap_safe={ri.overlap_safe}  core_path_found={ri.core_path_found}  → {outcome}"
                )
            if self.phase15_alarm_level and self.phase15_alarm_level != self.alarm_level:
                lines.append(f"  Phase 1.5 alarm: {self.alarm_level} → {self.phase15_alarm_level}")

        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Flat dict for CSV serialisation."""
        d = {
            "result": self.overall_pass,
            "valid_phrase": self.valid_phrase,
            "alarm_level": self.alarm_level,
            "phase15_alarm_level": self.phase15_alarm_level or self.alarm_level,
            "fail_locations": ";".join(self.fail_locations),
            "rel_matched": self.relationship.matched_text if self.relationship else "",
            "rel_distance": round(self.relationship.distance, 6) if self.relationship else "",
            "rel_status": self.relationship.status if self.relationship else "",
            "origin_matched": self.origin.matched_text if self.origin else "",
            "origin_distance": round(self.origin.distance, 6) if self.origin else "",
            "origin_status": self.origin.status if self.origin else "",
            "origin_candidates": ";".join(self.origin.candidates) if self.origin else "",
            "target_matched": self.target.matched_text if self.target else "",
            "target_distance": round(self.target.distance, 6) if self.target else "",
            "target_status": self.target.status if self.target else "",
            "target_candidates": ";".join(self.target.candidates) if self.target else "",
        }
        if self.modifier_retries:
            ri = self.modifier_retries[0]
            d["modifier_stripped"] = True
            d["modifier_component"] = ri.component
            d["modifier_text"] = ri.modifier
            d["modifier_core"] = ri.core_text
            d["modifier_category"] = ri.category
            d["modifier_overlap_safe"] = ri.overlap_safe
            d["modifier_core_path_found"] = ri.core_path_found
            d["modifier_alarm_upgraded"] = (
                ri.category == "staging"
                and ri.overlap_safe
                and ri.core_path_found
                and self.alarm_level == "FAIL"
            )
        else:
            d["modifier_stripped"] = False
            d["modifier_component"] = ""
            d["modifier_text"] = ""
            d["modifier_core"] = ""
            d["modifier_category"] = ""
            d["modifier_overlap_safe"] = False
            d["modifier_core_path_found"] = False
            d["modifier_alarm_upgraded"] = False
        return d
