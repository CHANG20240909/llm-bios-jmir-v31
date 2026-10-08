# NYU OLAB, 2023

"""
Inference-time screening defense against harm using ontology-enhanced review of medical named entities.

This module contains the main active defense function, which consumes a medical phrase or 'triplet' (origin, relationship, target)
and returns a boolean indicating whether the triplet is valid or not, according to the provided knowledge graph ground truth.
"""

# imports
import os
import time

import omegaconf
import pandas as pd
from tqdm import tqdm

from utils import *
from vector_db import VectorDatabase, KnowledgeGraph


def verify_medical_phrase(
    vector_db: VectorDatabase,
    knowledge_graph: KnowledgeGraph,
    phrase: MedicalPhrase,
    k_r: int = 1,
    k_c: int = 1,
    check_negatives: bool = False,
    check_exact_relationships: bool = False,
    check_exact_concepts: bool = False,
    negatives_pipeline=None,
    alarm_mode: bool = False,
    n_candidates: int = 3,
    strict_threshold: float = 0.05,
    loose_threshold: float = 0.3,
):
    """
    Score a medical phrase (origin, relationship, target) against the Knowledge Graph.

    When alarm_mode=False (default): returns (bool, str) for backward compatibility.
    When alarm_mode=True: returns VerificationResult with per-component distance
    tracking, alarm level, and fail locations for expert review.
    """

    # 1. Match phrase.relationship
    matched_relationships_query = vector_db.match_relationship(
        phrase.relationship, vector_db.relationship_collection.count()
    )
    matched_relationships_tentative = matched_relationships_query["documents"][0]
    matched_relationship_metadatas_tentative = matched_relationships_query["metadatas"][0]
    matched_relationship_distances_tentative = matched_relationships_query.get("distances", [[]])[0]

    matched_relationships = []
    matched_relationship_metadatas = []
    matched_relationship_distances = []
    filter_idx = 0

    while (
        filter_idx < len(matched_relationships_tentative)
        and len(matched_relationships) < k_r
    ):
        relationship = matched_relationships_tentative[filter_idx]
        relationship_metadata = matched_relationship_metadatas_tentative[filter_idx]
        rel_dist = matched_relationship_distances_tentative[filter_idx] if filter_idx < len(matched_relationship_distances_tentative) else float("inf")

        if relationship not in [r for r in matched_relationships]:
            matched_relationships.append(relationship)
            matched_relationship_metadatas.append(relationship_metadata)
            matched_relationship_distances.append(rel_dist)

        filter_idx += 1

    if check_exact_relationships:
        matched_relationships, matched_relationship_metadatas = relationship_in_list(
            phrase.relationship, matched_relationships, matched_relationship_metadatas
        )

    if check_negatives:
        positive_relationship = is_positive(phrase.relationship, negatives_pipeline)
    else:
        positive_relationship = True

    result = False
    valid_phrase = ""
    best_rel_dist = matched_relationship_distances[0] if matched_relationship_distances else float("inf")
    best_origin_dist = float("inf")
    best_target_dist = float("inf")
    best_origin_match = ""
    best_target_match = ""
    best_rel_match = matched_relationships[0] if matched_relationships else ""
    origin_cands = []
    origin_cand_dists = []
    target_cands = []
    target_cand_dists = []

    for rel_idx, (matched_relationship, matched_relationship_metadata) in enumerate(
        zip(matched_relationships, matched_relationship_metadatas)
    ):
        n_concept = max(k_c, n_candidates) if alarm_mode else k_c

        origin_query = vector_db.match_concept(
            phrase.origin, n_concept, matched_relationship_metadata["origins"],
        )
        origin_docs = origin_query["documents"][0]
        origin_dists = origin_query.get("distances", [[]])[0]

        if alarm_mode and rel_idx == 0:
            origin_cands = origin_docs[:n_candidates]
            origin_cand_dists = origin_dists[:n_candidates]
            if origin_dists:
                best_origin_dist = origin_dists[0]
                best_origin_match = origin_docs[0]

        matched_origins = origin_docs[:k_c]
        if check_exact_concepts:
            matched_origins = concept_in_list(phrase.origin, matched_origins)

        target_query = vector_db.match_concept(
            phrase.target, n_concept, matched_relationship_metadata["targets"],
        )
        target_docs = target_query["documents"][0]
        target_dists = target_query.get("distances", [[]])[0]

        if alarm_mode and rel_idx == 0:
            target_cands = target_docs[:n_candidates]
            target_cand_dists = target_dists[:n_candidates]
            if target_dists:
                best_target_dist = target_dists[0]
                best_target_match = target_docs[0]

        matched_targets = target_docs[:k_c]
        if check_exact_concepts:
            matched_targets = concept_in_list(phrase.target, matched_targets)

        result, valid_phrase = knowledge_graph.query(
            matched_origins, matched_targets, matched_relationship
        )

        if result:
            break
        else:
            valid_phrase = ""

    if not positive_relationship:
        result = not result

    if not alarm_mode:
        return result, valid_phrase

    rel_detail = MatchDetail(
        input_text=phrase.relationship,
        matched_text=best_rel_match,
        distance=best_rel_dist,
        status=classify_distance(best_rel_dist, strict_threshold, loose_threshold),
        candidates=[best_rel_match],
        candidate_distances=[best_rel_dist],
    )
    origin_detail = MatchDetail(
        input_text=phrase.origin,
        matched_text=best_origin_match,
        distance=best_origin_dist,
        status=classify_distance(best_origin_dist, strict_threshold, loose_threshold),
        candidates=origin_cands,
        candidate_distances=origin_cand_dists,
    )
    target_detail = MatchDetail(
        input_text=phrase.target,
        matched_text=best_target_match,
        distance=best_target_dist,
        status=classify_distance(best_target_dist, strict_threshold, loose_threshold),
        candidates=target_cands,
        candidate_distances=target_cand_dists,
    )

    fail_locs = []
    if not result:
        for label, detail in [("relationship", rel_detail), ("origin", origin_detail), ("target", target_detail)]:
            if detail.status in ("WARN", "FAIL"):
                fail_locs.append(label)
        if not fail_locs:
            fail_locs.append("path")

    if result:
        alarm = "PASS"
    else:
        statuses = [origin_detail.status, target_detail.status, rel_detail.status]
        if "FAIL" in statuses:
            alarm = "FAIL"
        elif "WARN" in statuses:
            alarm = "WARN"
        else:
            alarm = "FAIL"

    vr = VerificationResult(
        overall_pass=result,
        alarm_level=alarm,
        relationship=rel_detail,
        origin=origin_detail,
        target=target_detail,
        path_exists=result,
        valid_phrase=valid_phrase,
        fail_locations=fail_locs,
    )
    return vr


def try_modifier_retry(
    vr: VerificationResult,
    vector_db: VectorDatabase,
    knowledge_graph: KnowledgeGraph,
    phrase: MedicalPhrase,
    overlap_threshold: float = 0.3,
    n_candidates: int = 3,
) -> VerificationResult:
    """
    Phase 1.5: modifier-stripped retry for WARN/FAIL cases.

    For each WARN/FAIL component (origin or target), attempts to detect a medical
    modifier prefix, strip it, and re-search with the core concept.

    Alarm upgrade rules (see plan.md Phase 2A-EXT):
    - staging + overlap_safe + core_path_found + Phase 1 was FAIL → FAIL→WARN
    - genetic modifier: alarm locked, path info forwarded to Phase 2 only
    - overlap_safe=False: retry rejected (modifier is integral to disease name)
    """
    modifier_retries = []

    for component in ("origin", "target"):
        detail = getattr(vr, component)
        if detail is None or detail.status not in ("WARN", "FAIL"):
            continue

        result = extract_modifier_and_core(detail.input_text)
        if result is None:
            continue

        modifier, core_text, category = result

        # Re-search core concept (broad — no UID filter)
        core_query = vector_db.match_concept(core_text, n_candidates)
        core_candidates = core_query["documents"][0] if core_query["documents"] else []
        core_distances = core_query.get("distances", [[]])[0]
        core_distance = core_distances[0] if core_distances else float("inf")
        core_matched = core_candidates[0] if core_candidates else ""

        # Semantic overlap check (Phase 1.5c safety guard)
        overlap_safe = compute_candidate_overlap(
            detail.candidates, core_candidates, overlap_threshold
        )

        # Try KG path with core concept substituted
        core_path_found = False
        if overlap_safe:
            rel_query = vector_db.match_relationship(phrase.relationship, 1)
            if rel_query["documents"] and rel_query["documents"][0]:
                matched_rel = rel_query["documents"][0][0]
                rel_meta = rel_query["metadatas"][0][0] if rel_query["metadatas"][0] else {}
                origin_uid = rel_meta.get("origins", "")
                target_uid = rel_meta.get("targets", "")

                test_origin = core_text if component == "origin" else phrase.origin
                test_target = core_text if component == "target" else phrase.target

                orig_q = vector_db.match_concept(test_origin, 1, origin_uid)
                targ_q = vector_db.match_concept(test_target, 1, target_uid)
                matched_origins = orig_q["documents"][0][:1] if orig_q["documents"] else []
                matched_targets = targ_q["documents"][0][:1] if targ_q["documents"] else []

                path_result, _ = knowledge_graph.query(
                    matched_origins, matched_targets, matched_rel
                )
                core_path_found = path_result

        retry_info = ModifierRetryInfo(
            component=component,
            original_text=detail.input_text,
            modifier=modifier,
            core_text=core_text,
            category=category,
            meaning_sensitive=(category == "genetic"),
            core_distance=core_distance,
            core_matched=core_matched,
            core_path_found=core_path_found,
            overlap_safe=overlap_safe,
        )
        modifier_retries.append(retry_info)

    vr.modifier_retries = modifier_retries

    # Determine Phase 1.5 alarm level
    upgraded = any(
        ri.category == "staging"
        and ri.overlap_safe
        and ri.core_path_found
        for ri in modifier_retries
    )
    if upgraded and vr.alarm_level == "FAIL":
        vr.phase15_alarm_level = "WARN"
    else:
        vr.phase15_alarm_level = vr.alarm_level

    return vr


def defense(config, vector_db, knowledge_graph, medical_phrases):
    if config.verify.check_negatives:
        negative_pipeline = load_negative_pipeline()
    else:
        negative_pipeline = None

    alarm_mode = config.get("alarm_mode", False)
    strict_threshold = config.get("strict_threshold", 0.05)
    loose_threshold = config.get("loose_threshold", 0.3)
    n_candidates = config.get("n_candidates", 3)
    modifier_separation = config.get("modifier_separation", False)
    overlap_threshold = config.get("modifier_overlap_threshold", 0.3)

    # -- checkpoint / resume ---------------------------------------------------
    checkpoint_interval = config.get("checkpoint_interval", 100)
    output_stem = config.medical_phrases[:-4] if config.medical_phrases.endswith(".csv") else config.medical_phrases
    checkpoint_path = output_stem + "_checkpoint.csv"
    start_idx = 0

    if os.path.exists(checkpoint_path):
        saved = pd.read_csv(checkpoint_path)
        if "_checkpoint_idx" in saved.columns:
            start_idx = int(saved["_checkpoint_idx"].max()) + 1
            for col in saved.columns:
                if col != "_checkpoint_idx" and col in medical_phrases.columns:
                    continue
                if col == "_checkpoint_idx":
                    continue
                medical_phrases[col] = None
            for col in saved.columns:
                if col == "_checkpoint_idx":
                    continue
                medical_phrases.loc[: start_idx - 1, col] = saved.loc[: start_idx - 1, col]
            print(
                f"[checkpoint] Resuming from row {start_idx}/{len(medical_phrases)}",
                flush=True,
            )

    # -- main loop -------------------------------------------------------------
    total = len(medical_phrases)
    t0 = time.time()
    log_interval = max(1, total // 20)

    for row_idx in tqdm(range(start_idx, total), initial=start_idx, total=total):
        row = medical_phrases.iloc[row_idx]
        phrase = MedicalPhrase(row["origin"], row["relationship"], row["target"])

        out = verify_medical_phrase(
            vector_db, knowledge_graph, phrase,
            config.verify.k_r, config.verify.k_c,
            config.verify.check_negatives,
            config.verify.check_exact_relationships,
            config.verify.check_exact_concepts,
            negative_pipeline,
            alarm_mode=alarm_mode,
            n_candidates=n_candidates,
            strict_threshold=strict_threshold,
            loose_threshold=loose_threshold,
        )

        if alarm_mode:
            vr = out

            if modifier_separation and vr.alarm_level in ("WARN", "FAIL"):
                vr = try_modifier_retry(
                    vr, vector_db, knowledge_graph, phrase,
                    overlap_threshold=overlap_threshold,
                    n_candidates=n_candidates,
                )
            else:
                vr.phase15_alarm_level = vr.alarm_level

            medical_phrases.loc[row_idx, "result"] = vr.overall_pass
            medical_phrases.loc[row_idx, "valid_phrase"] = vr.valid_phrase
            for col, val in vr.to_dict().items():
                if col not in ("result", "valid_phrase"):
                    medical_phrases.loc[row_idx, col] = val

            icon = {"PASS": "[OK]", "WARN": "[??]", "FAIL": "[!!]"}
            p15_info = ""
            if vr.phase15_alarm_level and vr.phase15_alarm_level != vr.alarm_level:
                p15_info = f"  phase1.5={vr.phase15_alarm_level} [UPGRADED]"
            print(
                f"{icon.get(vr.alarm_level, '[??]')} {phrase.origin} -- {phrase.relationship} -- {phrase.target}"
                f"  alarm={vr.alarm_level}{p15_info}  pass={vr.overall_pass}",
                flush=True,
            )
        else:
            result, valid_phrases = out
            medical_phrases.loc[row_idx, "result"] = result
            medical_phrases.loc[row_idx, "valid_phrase"] = valid_phrases
            label = "Non-harmful" if result else "Harmful"
            print(
                f"Phrase: {phrase.origin} -- {phrase.relationship} -- {phrase.target} -- Result: {label}",
                flush=True,
            )

        # -- progress logging --------------------------------------------------
        done = row_idx - start_idx + 1
        if done % log_interval == 0 or row_idx == total - 1:
            elapsed = time.time() - t0
            throughput = done / elapsed if elapsed > 0 else 0
            remaining = (total - row_idx - 1) / throughput if throughput > 0 else 0
            print(
                f"[defense] {row_idx + 1}/{total} "
                f"({elapsed:.0f}s elapsed, {throughput:.2f} phrases/s, ~{remaining:.0f}s left)",
                flush=True,
            )

        # -- periodic checkpoint -----------------------------------------------
        if checkpoint_interval > 0 and done % checkpoint_interval == 0:
            cp = medical_phrases.copy()
            cp["_checkpoint_idx"] = range(len(cp))
            cp.to_csv(checkpoint_path, index=False)

    # clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return medical_phrases


def setup_kg_vector_db(config):
    gt_path = config.knowledge_graph.ground_truth
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"[KG] ground_truth file not found: {gt_path}")

    streaming = config.knowledge_graph.get("streaming_kg", False)
    if streaming:
        print("[KG] streaming_kg=True. Loading JSONL as generator (low-memory mode).", flush=True)
        ground_truth = json_io_stream(gt_path)
    else:
        ground_truth = json_io(gt_path)

    knowledge_graph = KnowledgeGraph(ground_truth)

    rebuild_vector_db = config.knowledge_graph.get("rebuild_vector_db", False)
    print(f"[KG] rebuild_vector_db={rebuild_vector_db}", flush=True)

    vector_db = VectorDatabase(
        config.knowledge_graph.vector_db_root,
        knowledge_graph.relationships,
        knowledge_graph.concepts,
        config.knowledge_graph.embedding_model,
        config.knowledge_graph.embedding_batch_size,
        rebuild_vector_db,
    )

    return knowledge_graph, vector_db


if __name__ == "__main__":
    # Args
    args = parse_args()

    # Load config
    config = omegaconf.OmegaConf.load(args.config)

    # Pretty print config
    pretty_print_omegaconf(config)

    # Set seed
    set_seeds(config.seed)

    # load knowledge graph and vector database
    global knowledge_graph, vector_db
    knowledge_graph, vector_db = setup_kg_vector_db(config)

    # Load medical phrases dataframe
    medical_phrases = pd.read_csv(config.medical_phrases)

    # Check if "result" and "valid_phrase" columns exist, add blank columns if not
    if "result" not in medical_phrases.columns:
        medical_phrases["result"] = False

    if "valid_phrase" not in medical_phrases.columns:
        medical_phrases["valid_phrase"] = ""

    # Main
    medical_phrases = defense(config, vector_db, knowledge_graph, medical_phrases)

    # Save medical phrases dataframe
    medical_phrases.to_csv(config.medical_phrases[:-4] + "_scored.csv", index=False)
