"""The orchestration core: decide a ``New Account`` value for every row.

Decision priority (highest wins):
    1. Existing seed value in the upload   -> kept as-is.
    2. A matching user keyword rule         -> deterministic, very high confidence.
    3. A previously-approved learned mapping (exact text signature).
    4. A high-confidence **ML** prediction  -> only when a model is trained and
       the active mode allows it.
    5. **Similarity** propagation from seed rows (the reliable foundation).
    6. Nothing -> left blank (review queue or no-match).

The ML layer is strictly additive: if it is disabled, untrained, or errors out,
every decision falls through to the similarity engine exactly as before. Each
output row records ``engine_used``, ``confidence`` and ``rationale``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from models.schemas import (
    FillAction,
    FillResult,
    FillSource,
    RunSummary,
    TransactionGroup,
)
from src.config import Config
from src.data_loader import SIM_TEXT_COL, LoadedData, _has_value
from src.rules_manager import RulesManager
from src.similarity import Embedder, cosine_sim_matrix, group_transactions
from utils.account_codes import normalize_code
from utils.logging_setup import get_logger

logger = get_logger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]

# Modes the engine understands internally. "assist" is the conservative
# behaviour used by "auto" when there are few labelled examples.
_ML_MODES = {"assist", "hybrid", "prefer_ml"}


@dataclass
class EngineResult:
    """Bundle of everything a run produces."""

    df: pd.DataFrame                       # dataframe with New Account filled in
    results: List[FillResult]              # per-row decisions
    groups: List[TransactionGroup]         # discovered similarity groups
    summary: RunSummary
    backend: str = ""
    mode: str = "similarity_only"          # effective decision mode used
    embeddings: Optional[np.ndarray] = field(default=None, repr=False)

    def review_indices(self) -> List[int]:
        """Rows the user should look at (flagged fills + needs-review)."""
        return [
            r.row_index
            for r in self.results
            if r.action in (FillAction.FILLED_REVIEW, FillAction.NEEDS_REVIEW)
        ]


class FillDownEngine:
    """Runs the full fill-down pipeline on a :class:`LoadedData`.

    ``model_manager`` and ``mode`` are optional. When omitted (or when no model
    is trained / ML is disabled) the engine uses similarity only.
    """

    def __init__(
        self,
        config: Config,
        rules_manager: RulesManager,
        learned_lookup: Optional[Dict[str, str]] = None,
        model_manager=None,
        mode: Optional[str] = None,
        client_id: Optional[str] = None,
        blocked_lookup: Optional[Dict[str, set]] = None,
    ):
        self.config = config
        self.rules = rules_manager
        self.learned_lookup = learned_lookup or {}
        self.model_manager = model_manager
        self.mode = mode  # raw user choice; None -> config.ml.mode
        self.client_id = client_id
        # signature -> set of account codes the user has rejected for it.
        self.blocked_lookup = blocked_lookup or {}
        self._insight_cache: Dict[str, str] = {}

    # --------------------------------------------------------- mode resolve
    def _effective_mode(self) -> str:
        """Resolve the active decision mode, defaulting safely to similarity."""
        if not self.config.ml.enabled:
            return "similarity_only"
        if self.model_manager is None or not self.model_manager.has_model():
            return "similarity_only"
        choice = (self.mode or self.config.ml.mode or "auto").lower()
        if choice == "auto":
            try:
                return self.model_manager.progressive_mode()
            except Exception:  # noqa: BLE001
                return "similarity_only"
        if choice in _ML_MODES or choice == "similarity_only":
            return choice
        return "similarity_only"

    # --------------------------------------------------------------- public
    def run(self, data: LoadedData, progress_cb: ProgressCB = None) -> EngineResult:
        df = data.df.copy()
        na_col = data.new_account_col
        texts: List[str] = df[SIM_TEXT_COL].tolist()
        n = len(df)

        def report(frac: float, msg: str) -> None:
            if progress_cb:
                progress_cb(frac, msg)

        report(0.05, "Preparing transactions…")
        effective_mode = self._effective_mode()

        # --- 1. embeddings + grouping (similarity foundation) -----------
        embedder = Embedder(self.config, progress_cb=lambda m: report(0.15, m))
        grouping = group_transactions(
            texts, self.config, embedder=embedder,
            progress_cb=lambda m: report(0.35, m),
        )
        labels = grouping.labels
        embeddings = grouping.embeddings

        # Identify seed rows (already have a value).
        seed_mask = df[na_col].apply(_has_value).to_numpy()
        seed_indices = np.where(seed_mask)[0]
        seed_codes = {int(i): str(df.iloc[i][na_col]).strip() for i in seed_indices}

        # --- 2. optional ML predictions for blank rows ------------------
        ml_predictions: Dict[int, object] = {}
        if effective_mode in _ML_MODES and self.model_manager is not None:
            report(0.45, "Consulting trained ML model…")
            blank_idx = [i for i in range(n) if not seed_mask[i]]
            try:
                preds = self.model_manager.predict([texts[i] for i in blank_idx])
                ml_predictions = dict(zip(blank_idx, preds))
            except Exception as exc:  # noqa: BLE001 - never break on ML
                logger.warning("ml_predict_failed", error=str(exc))
                effective_mode = "similarity_only"

        report(0.55, "Scoring and propagating account codes…")
        active_rules = self.rules.list_rules(enabled_only=True)
        results: List[FillResult] = []

        # Precompute similarity of every row to every seed row (if any seeds).
        sim_to_seeds = None
        if len(seed_indices) > 0 and embeddings.shape[0] == n:
            seed_emb = embeddings[seed_indices]
            sim_to_seeds = np.clip(embeddings @ seed_emb.T, -1.0, 1.0)

        na_loc = df.columns.get_loc(na_col)
        for i in range(n):
            row = df.iloc[i]
            text = texts[i]
            group_id = int(labels[i]) if i < len(labels) else -1
            group_id = group_id if group_id != -1 else None

            # 1) Existing seed -> keep.
            if seed_mask[i]:
                results.append(FillResult(
                    row_index=i, original_value=seed_codes[i],
                    proposed_value=seed_codes[i], confidence=1.0,
                    source=FillSource.SEED, engine_used="seed",
                    action=FillAction.KEPT_SEED, group_id=group_id,
                    rationale="Value already present in upload (seed).",
                ))
                continue

            # 2) Keyword rule.
            match = self.rules.match_row(text, row=row, rules=active_rules)
            if match:
                code = normalize_code(match.account_code)
                df.iat[i, na_loc] = code
                rationale = f"Matched rule '{match.rule.keyword}' -> {code}."
                # Surface rule-vs-memory conflicts: the rule still wins (cascade
                # order), but the user deserves to see the disagreement.
                remembered = self.learned_lookup.get(text)
                if remembered:
                    remembered_code = normalize_code(remembered)
                    if remembered_code and remembered_code != code:
                        rationale += (f" Learned memory had '{remembered_code}' "
                                      "— your rule takes precedence.")
                results.append(FillResult(
                    row_index=i, original_value="", proposed_value=code,
                    confidence=self.config.confidence.rule_match_confidence,
                    source=FillSource.RULE, engine_used="rules",
                    action=FillAction.AUTO_FILLED, group_id=group_id,
                    rationale=rationale,
                ))
                continue

            # 3) Learned mapping (exact signature) — unless the user rejected
            #    this exact text -> code pairing before.
            if text in self.learned_lookup:
                code = normalize_code(self.learned_lookup[text])
                if code in self.blocked_lookup.get(text, set()):
                    results.append(FillResult(
                        row_index=i, original_value="", proposed_value=None,
                        confidence=0.0, source=FillSource.NONE,
                        engine_used="none", action=FillAction.NEEDS_REVIEW,
                        group_id=group_id,
                        rationale=(f"You previously rejected '{code}' for this "
                                   "exact transaction — please choose the "
                                   "right code."),
                    ))
                    continue
                df.iat[i, na_loc] = code
                results.append(FillResult(
                    row_index=i, original_value="", proposed_value=code,
                    confidence=self.config.confidence.learned_match_confidence,
                    source=FillSource.LEARNED, engine_used="learned",
                    action=FillAction.AUTO_FILLED, group_id=group_id,
                    rationale="Matched a previously approved transaction.",
                ))
                continue

            # 4/5) Similarity decision, optionally overridden by ML.
            sim_res = self._similarity_decision(
                i, sim_to_seeds, seed_indices, seed_codes, group_id,
                text=text)
            decision = self._combine_with_ml(
                sim_res, ml_predictions.get(i), effective_mode, group_id)

            if decision.action in (FillAction.AUTO_FILLED, FillAction.FILLED_REVIEW):
                df.iat[i, na_loc] = normalize_code(decision.proposed_value or "")
            results.append(decision)

        report(0.9, "Building groups and summary…")
        groups = self._build_groups(df, na_col, texts, labels, embeddings, seed_codes)
        summary = self._summarise(data, results, grouping.backend, len(groups))
        summary.notes = f"mode={effective_mode}"

        report(1.0, "Done.")
        logger.info("run_complete", rows=summary.total_rows,
                    auto_filled=summary.auto_filled,
                    filled_review=summary.filled_review,
                    needs_review=summary.needs_review,
                    no_match=summary.no_match, backend=summary.embedding_backend,
                    mode=effective_mode)

        return EngineResult(df=df, results=results, groups=groups, summary=summary,
                            backend=grouping.backend, mode=effective_mode,
                            embeddings=embeddings)

    # --------------------------------------------------- rule-driven (only)
    def run_selected_rules(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        rules=None,
        client_id: Optional[str] = None,
    ):
        """Deterministic rule-only pass over ``df`` — the reliable primary path.

        Delegates to :meth:`RulesManager.apply_rules_to_dataframe`, so it:
          * never overwrites a row whose ``New Account`` is already set, and
          * only fills blank rows, in rule-priority order.

        Returns ``(df, results)`` where ``results`` is a list of
        :class:`RuleRunResult` (row_index, original/proposed value, engine_used,
        confidence, rationale, matched_rule_name, matched_pattern, status).
        """
        if rules is None:
            rules = self.rules.list_rules(enabled_only=True, client_id=client_id)
        out_df, results = self.rules.apply_rules_to_dataframe(
            df, rules=rules, text_columns=text_columns)
        # Reinforce + explain using the account knowledge base (best-effort).
        try:
            self.rules.reinforce_from_rule_run(
                results, out_df, text_columns, client_id=client_id)
            self.rules.enrich_results(results, client_id=client_id)
        except Exception as exc:  # noqa: BLE001 - never break a rule run
            logger.warning("account_learning_run_failed", error=str(exc))
        return out_df, results

    @staticmethod
    def get_rule_application_summary(results) -> dict:
        """Metrics for a rule run: protected / filled / blank + success rate."""
        from src.rules_manager import get_rule_application_summary as _summary
        return _summary(results)

    # ------------------------------------------------------- similarity step
    def _account_context(self, code: str) -> str:
        """Short learned/glossary context for a code, cached per run.

        Powers rationales like "looks like 6322 Call Center Services — learned
        patterns for 6322: cunningham, call center (seen in 18 approved rows)".
        Best-effort: empty string when nothing is known.
        """
        code = (code or "").strip()
        if not code:
            return ""
        if code in self._insight_cache:
            return self._insight_cache[code]
        parts: List[str] = []
        try:
            entry = self.config.glossary_entry(code)
            name = str(entry.get("name", "")).strip() if entry else ""
            if name:
                parts.append(name)
        except Exception:  # noqa: BLE001 - glossary is best-effort
            pass
        try:
            insight = self.rules.account_insight(code, self.client_id)
            if insight:
                parts.append(insight)
            prof = self.rules.get_account_profile(code, self.client_id)
            usage = int(prof.get("usage_count", 0)) if prof else 0
            if usage:
                parts.append(f"seen in {usage} approved row"
                             f"{'s' if usage != 1 else ''}")
        except Exception:  # noqa: BLE001 - learning context is best-effort
            pass
        note = " — ".join(parts)
        self._insight_cache[code] = note
        return note

    def _similarity_decision(
        self,
        i: int,
        sim_to_seeds: Optional[np.ndarray],
        seed_indices: np.ndarray,
        seed_codes: Dict[int, str],
        group_id: Optional[int],
        text: str = "",
    ) -> FillResult:
        conf_cfg = self.config.confidence
        thr = self.config.similarity.similarity_threshold

        if sim_to_seeds is None or len(seed_indices) == 0:
            return FillResult(
                row_index=i, original_value="", proposed_value=None,
                confidence=0.0, source=FillSource.NONE, engine_used="none",
                action=FillAction.NO_MATCH, group_id=group_id,
                rationale="No seed examples available to learn from.",
            )

        sims = sim_to_seeds[i]
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])
        best_seed_row = int(seed_indices[best_pos])
        proposed = seed_codes[best_seed_row]

        similar_pos = np.where(sims >= thr)[0]
        if len(similar_pos) > 0:
            codes = [seed_codes[int(seed_indices[p])] for p in similar_pos]
            agreement = codes.count(proposed) / len(codes)
        else:
            codes = [proposed]
            agreement = 1.0

        confidence = float(np.clip(best_sim * (0.6 + 0.4 * agreement), 0.0, 1.0))
        rationale = (
            f"Most similar seed (cos={best_sim:.2f}) is '{proposed}'. "
            f"Agreement among {len(similar_pos)} similar seeds: {agreement:.0%}."
        )
        context = self._account_context(proposed)
        if context:
            rationale += f" Looks like {context}."

        # A pairing the user explicitly rejected must not come back as a fill.
        if text and proposed in self.blocked_lookup.get(text, set()):
            return FillResult(
                row_index=i, original_value="", proposed_value=proposed,
                confidence=confidence, source=FillSource.SIMILARITY,
                engine_used="similarity", action=FillAction.NEEDS_REVIEW,
                group_id=group_id,
                rationale=(f"Similar rows suggest '{proposed}', but you "
                           "rejected that pairing before — please choose the "
                           "right code."),
            )

        # Seed disagreement: when the nearest seeds vote for *different*
        # accounts, never auto-fill — send to review with the vote split.
        distinct = sorted(set(codes))
        if len(distinct) >= 2:
            from collections import Counter
            votes = Counter(codes)
            split = ", ".join(f"{c} ×{n}" for c, n in votes.most_common())
            split_note = (f"Nearest seeds disagree ({split}). ")
            if agreement < 0.5:
                return FillResult(
                    row_index=i, original_value="", proposed_value=proposed,
                    confidence=confidence, source=FillSource.SIMILARITY,
                    engine_used="similarity", action=FillAction.NEEDS_REVIEW,
                    group_id=group_id,
                    rationale=(split_note + rationale +
                               " Not auto-filled — please review."),
                )
            if agreement < 0.75 and confidence >= conf_cfg.auto_apply_cutoff:
                # Weak majority: fill, but always flag for review.
                return FillResult(
                    row_index=i, original_value="", proposed_value=proposed,
                    confidence=confidence, source=FillSource.SIMILARITY,
                    engine_used="similarity", action=FillAction.FILLED_REVIEW,
                    group_id=group_id,
                    rationale=(split_note + rationale +
                               " Weak majority — flagged for review."),
                )

        if confidence >= conf_cfg.auto_apply_cutoff:
            action = FillAction.AUTO_FILLED
        elif confidence >= conf_cfg.review_cutoff:
            action = FillAction.FILLED_REVIEW
        elif best_sim >= conf_cfg.review_cutoff * 0.7:
            return FillResult(
                row_index=i, original_value="", proposed_value=proposed,
                confidence=confidence, source=FillSource.SIMILARITY,
                engine_used="similarity", action=FillAction.NEEDS_REVIEW,
                group_id=group_id,
                rationale=rationale + " Below confidence cutoff — please review.",
            )
        else:
            return FillResult(
                row_index=i, original_value="", proposed_value=None,
                confidence=confidence, source=FillSource.NONE, engine_used="none",
                action=FillAction.NO_MATCH, group_id=group_id,
                rationale="No sufficiently similar seed transaction found.",
            )

        return FillResult(
            row_index=i, original_value="", proposed_value=proposed,
            confidence=confidence, source=FillSource.SIMILARITY,
            engine_used="similarity", action=action, group_id=group_id,
            rationale=rationale,
        )

    # ------------------------------------------------------------- ML layer
    def _combine_with_ml(self, sim_res: FillResult, ml_pred, mode: str,
                         group_id: Optional[int]) -> FillResult:
        """Layer an ML prediction on top of the similarity result.

        Returns ``sim_res`` unchanged whenever ML is unavailable/low-confidence,
        guaranteeing the similarity engine remains the safe default.
        """
        if mode == "similarity_only" or ml_pred is None:
            return sim_res
        ml_label = getattr(ml_pred, "label", None)
        ml_conf = float(getattr(ml_pred, "confidence", 0.0) or 0.0)
        ml_name = getattr(ml_pred, "model_name", "ml") or "ml"
        if not ml_label or ml_conf <= 0.0:
            return sim_res

        conf_cfg = self.config.confidence
        ml_cut = self.config.ml.ml_confidence_cutoff
        sim_label = sim_res.proposed_value
        i = sim_res.row_index
        agree = bool(sim_label) and sim_label == ml_label

        def mk(proposed, conf, action, engine, rationale,
               source=FillSource.ML) -> FillResult:
            return FillResult(
                row_index=i, original_value="", proposed_value=proposed,
                confidence=float(np.clip(conf, 0.0, 1.0)), source=source,
                engine_used=engine, action=action, group_id=group_id,
                rationale=rationale)

        ml_engine = f"ml:{ml_name}"
        high = ml_conf >= ml_cut

        # ---- high-confidence ML cases shared by all ML modes -----------
        if high and agree:
            return mk(ml_label, max(ml_conf, sim_res.confidence),
                      FillAction.AUTO_FILLED, "ml+similarity",
                      f"ML ({ml_name}, {ml_conf:.2f}) and similarity agree on "
                      f"'{ml_label}'.")
        if high and sim_label and not agree:
            return mk(ml_label, min(ml_conf, conf_cfg.auto_apply_cutoff),
                      FillAction.NEEDS_REVIEW, f"{ml_engine} vs similarity",
                      f"ML suggests '{ml_label}' ({ml_conf:.2f}) but similarity "
                      f"suggests '{sim_label}' — please review.")
        if high:  # confident ML, similarity had nothing
            return mk(ml_label, ml_conf, FillAction.AUTO_FILLED, ml_engine,
                      f"ML ({ml_name}) predicts '{ml_label}' with high "
                      f"confidence ({ml_conf:.2f}).")

        # ---- ML not high-confidence ------------------------------------
        if mode == "prefer_ml":
            if ml_conf >= conf_cfg.review_cutoff:
                return mk(ml_label, ml_conf, FillAction.FILLED_REVIEW, ml_engine,
                          f"ML ({ml_name}) predicts '{ml_label}' ({ml_conf:.2f}); "
                          "applied but flagged for review.")
            return sim_res  # fall back to similarity

        # assist + hybrid: similarity stays primary when ML is not confident.
        if sim_res.action in (FillAction.AUTO_FILLED, FillAction.FILLED_REVIEW):
            return sim_res
        if mode == "hybrid" and ml_conf >= conf_cfg.review_cutoff:
            if agree:
                return mk(ml_label, max(ml_conf, sim_res.confidence),
                          FillAction.FILLED_REVIEW, "ml+similarity",
                          f"ML and similarity lean to '{ml_label}' — review.")
            return mk(ml_label, ml_conf, FillAction.NEEDS_REVIEW, ml_engine,
                      f"ML ({ml_name}) tentatively suggests '{ml_label}' "
                      f"({ml_conf:.2f}) — please review.")
        return sim_res

    # --------------------------------------------------------------- groups
    def _build_groups(
        self,
        df: pd.DataFrame,
        na_col: str,
        texts: List[str],
        labels: np.ndarray,
        embeddings: np.ndarray,
        seed_codes: Dict[int, str],
    ) -> List[TransactionGroup]:
        groups: List[TransactionGroup] = []
        unique = sorted({int(l) for l in labels if l != -1})
        for gid in unique:
            members = [int(i) for i in np.where(labels == gid)[0]]
            seeds = [seed_codes[i] for i in members if i in seed_codes]
            avg_sim = 0.0
            if len(members) > 1 and embeddings.shape[0] == len(labels):
                sub = embeddings[members]
                mat = cosine_sim_matrix(sub)
                iu = np.triu_indices(len(members), k=1)
                if len(iu[0]) > 0:
                    avg_sim = float(np.mean(mat[iu]))
            rep = texts[members[0]] if members else ""
            groups.append(TransactionGroup(
                group_id=gid,
                row_indices=members,
                seed_account=(max(set(seeds), key=seeds.count) if seeds else None),
                seed_count=len(seeds),
                avg_similarity=round(avg_sim, 4),
                representative_text=rep[:120],
            ))
        return groups

    # -------------------------------------------------------------- summary
    def _summarise(
        self,
        data: LoadedData,
        results: List[FillResult],
        backend: str,
        n_groups: int,
    ) -> RunSummary:
        counts = {a: 0 for a in FillAction}
        for r in results:
            counts[r.action] += 1
        return RunSummary(
            file_name=data.source_name,
            total_rows=len(results),
            seeds=counts[FillAction.KEPT_SEED],
            auto_filled=counts[FillAction.AUTO_FILLED],
            filled_review=counts[FillAction.FILLED_REVIEW],
            needs_review=counts[FillAction.NEEDS_REVIEW],
            no_match=counts[FillAction.NO_MATCH],
            groups_found=n_groups,
            embedding_backend=backend,
            similarity_threshold=self.config.similarity.similarity_threshold,
            auto_apply_cutoff=self.config.confidence.auto_apply_cutoff,
        )
