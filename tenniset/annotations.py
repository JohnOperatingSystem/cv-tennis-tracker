"""Read TenniSet annotations and translate them into tracker-friendly labels.

TenniSet's generalized JSON files contain strong temporal labels for serves and
hits. Point winners are stored as player names in the raw files, so this module
uses the paired raw/generalized child events to translate a winner to the
camera-relative ``near`` or ``far`` side. Terminal reasons are weak labels
parsed from the corrected point descriptions and retain their confidence and
matched evidence so they can be reviewed instead of being treated as truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Iterable


EVENT_CLASSES = (
    "OTH",
    "SFI",
    "SFF",
    "SFL",
    "SNI",
    "SNF",
    "SNL",
    "HFL",
    "HFR",
    "HNL",
    "HNR",
)
REASON_CLASSES = ("net", "out", "winner")


@dataclass(frozen=True)
class WeakReason:
    reason: str
    coarse_reason: str | None
    confidence: float
    matched_text: str | None
    needs_review: bool

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TennisEvent:
    video_id: str
    event_id: str
    kind: str
    start_frame: int
    end_frame: int
    player_side: str
    event_class: str
    result: str | None = None
    stroke_side: str | None = None
    stroke_type: str | None = None

    @property
    def center_frame(self):
        return round((self.start_frame + self.end_frame) / 2)

    def to_dict(self):
        value = asdict(self)
        value["center_frame"] = self.center_frame
        return value


@dataclass(frozen=True)
class TennisPoint:
    video_id: str
    point_id: str
    local_id: str
    start_frame: int
    end_frame: int
    winner_side: str | None
    score: str
    description: str
    generalized_description: str
    terminal_reason: str
    coarse_reason: str | None
    reason_confidence: float
    reason_evidence: str | None
    needs_review: bool

    def to_dict(self):
        return asdict(self)


_REASON_RULES = (
    (
        "double_fault",
        "out",
        0.99,
        (
            r"\bdouble\s+fault\b",
        ),
    ),
    (
        "net",
        "net",
        0.96,
        (
            r"\bfails?\s+to\s+(?:clear|cross)\s+the\s+net\b",
            r"\b(?:fails?\s+to|doesn['’]?t|is\s+(?:un|not\s+)able\s+to)\s+(?:make\s+it\s+over|clear)\s+the\s+net\b",
            r"\b(?:hits?|catches?|finds?|clips?)\s+(?:the\s+)?net\b",
            r"\b(?:puts?|sends?|returns?|drives?|volleys?|hits?|hitting)\b[^.]{0,55}\binto\s+the\s+net\b",
            r"\bstruggles?\s+to\s+return\s+it\s+over\s+the\s+net\b",
            r"\b(?:goes?|flies?|return)\s+into\s+(?:the\s+){0,2}net\b",
            r"\b(?:it['’]?s|ball['’]?s)?\s*into\s+(?:the\s+)?net\b",
            r"\bmiss-hits?\b[^.]{0,35}\bat\s+(?:the\s+)?net\b",
            r"\blands?\s+at\s+the\s+net\b",
            r"\bnets?\s+(?:a|the|his|her|one|it|forehand|backhand)\b",
        ),
    ),
    (
        "out",
        "out",
        0.94,
        (
            r"\b(?:lands?|goes?|flies?|hits?|sends?|returns?|puts?|sprays?)\b[^.]{0,55}\b(?:outside|out(?:-side)?|wide|long)\b",
            r"\bfails?\s+to\s+(?:land|stay|keep)[^.]{0,45}\b(?:inside|in\s+the\s+play)\b",
            r"\b(?:doesn['’]?t|fails?\s+to)\s+land\s+(?:inside|in)\s+(?:the\s+)?court\b",
            r"\bfails?\s+to\s+land\s+it\s+in\s+(?:the\s+)?court\b",
            r"\bunable\s+to\s+keep\b[^.]{0,45}\bin\s+play\b",
            r"\bstruggles?\s+to\s+keep\b[^.]{0,55}\b(?:in\s+(?:a\s+)?rally|inside|in\s+(?:the\s+)?court)\b",
            r"\bmisses?\s+(?:the\s+)?(?:target|court)\b",
            r"\bover-cooks?\b",
            r"\b(?:outside|out(?:-side)?|wide|long)\s+(?:of\s+)?the\s+court\b",
            r"\bball\s+(?:is\s+)?out\b",
        ),
    ),
    (
        "ace",
        "winner",
        0.99,
        (
            r"\b(?:is\s+an?|an?)\s+ace\b",
            r"\baces?\b",
        ),
    ),
    (
        "winner",
        "winner",
        0.97,
        (
            r"\bwinner\b",
            r"\bwinning\s+(?:shot|return|forehand|backhand|volley)\b",
        ),
    ),
    (
        "unreturned",
        "winner",
        0.86,
        (
            r"\b(?:unable|fails?)\s+to\s+return\b",
            r"\bhas\s+no\s+answer\b",
            r"\bonly\s+reaches\s+to\s+it\b",
            r"\bonly\s+manages?\s+to\s+touch\s+it\b",
            r"\bfails?\s+to\s+(?:put|get)\s+it\s+back\b",
            r"\b(?:unable|struggles?)\s+to\s+get\s+it\s+back\b",
            r"\b(?:faces?|has)\s+difficulty\s+in\s+returning\s+it\b",
            r"\bstruggles?\s+to\s+put\s+it\s+back\b",
            r"\bmisses?\s+(?:a\s+)?(?:forehand|backhand|ls|rs)?\s*volley\b",
            r"\bvolley\s+is\s+a\s+miss-hit\b",
            r"\bstruggles?\s+to\s+execute\s+it\s+properly\b",
            r"\bfails?\s+to\s+control\s+it\b",
            r"\bstruggles\s+with\s+it\b",
        ),
    ),
)


def infer_terminal_reason(description: str) -> WeakReason:
    """Infer a reviewable terminal-reason label from corrected commentary."""
    normalized = " ".join(str(description or "").lower().split())
    let_cord_winner = re.search(
        r"\bwinner\b[^.]{0,50}\b(?:clips?|catches?)\s+(?:the\s+)?net\b",
        normalized,
    )
    if let_cord_winner:
        return WeakReason(
            reason="winner",
            coarse_reason="winner",
            confidence=0.97,
            matched_text=let_cord_winner.group(0),
            needs_review=False,
        )
    for reason, coarse_reason, confidence, patterns in _REASON_RULES:
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return WeakReason(
                    reason=reason,
                    coarse_reason=coarse_reason,
                    confidence=confidence,
                    matched_text=match.group(0),
                    needs_review=confidence < 0.90,
                )
    return WeakReason(
        reason="unknown",
        coarse_reason=None,
        confidence=0.0,
        matched_text=None,
        needs_review=True,
    )


class TenniSetAnnotations:
    """Load and validate the paired raw/generalized TenniSet annotations."""

    def __init__(self, annotation_dir, overrides_path=None):
        self.annotation_dir = Path(annotation_dir)
        self.generalized_dir = self.annotation_dir / "generalised"
        if not self.generalized_dir.is_dir():
            raise FileNotFoundError(
                f"Missing TenniSet generalized annotations: {self.generalized_dir}"
            )
        self._global_point_ids = self._read_global_point_ids()
        self._captions = self._read_captions()
        self.events: list[TennisEvent] = []
        self.points: list[TennisPoint] = []
        self.video_ids: list[str] = []
        self._load()
        default_overrides = self.annotation_dir / "reason_overrides.jsonl"
        override_source = (
            Path(overrides_path) if overrides_path is not None else default_overrides
        )
        if override_source.is_file():
            self._apply_overrides(override_source)

    def _load(self):
        generalized_paths = sorted(self.generalized_dir.glob("V*.json"))
        if not generalized_paths:
            raise FileNotFoundError(
                f"No V*.json files found in {self.generalized_dir}"
            )

        for generalized_path in generalized_paths:
            video_id = generalized_path.stem
            raw_path = self.annotation_dir / generalized_path.name
            if not raw_path.is_file():
                raise FileNotFoundError(f"Missing paired raw annotation: {raw_path}")
            generalized = self._read_json(generalized_path)
            raw = self._read_json(raw_path)
            self.video_ids.append(video_id)
            self.events.extend(self._load_video_events(video_id, generalized))
            self.points.extend(
                self._load_video_points(video_id, raw, generalized)
            )

        self.events.sort(key=lambda row: (row.video_id, row.start_frame, row.kind))
        self.points.sort(key=lambda row: (row.video_id, row.start_frame))

    def _apply_overrides(self, path):
        overrides = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid override JSON at {path}:{line_number}"
                ) from error
            point_id = str(row.get("point_id", "")).strip()
            if point_id:
                overrides[point_id] = row

        updated = []
        for point in self.points:
            override = overrides.get(point.point_id)
            if override is None:
                updated.append(point)
                continue
            coarse_reason = str(
                override.get("coarse_reason", point.coarse_reason) or ""
            ).lower()
            winner_side = str(
                override.get("winner_side", point.winner_side) or ""
            ).lower()
            if coarse_reason not in REASON_CLASSES:
                raise ValueError(
                    f"Invalid coarse_reason override for {point.point_id}: "
                    f"{coarse_reason!r}"
                )
            if winner_side not in ("near", "far"):
                raise ValueError(
                    f"Invalid winner_side override for {point.point_id}: "
                    f"{winner_side!r}"
                )
            updated.append(
                replace(
                    point,
                    winner_side=winner_side,
                    terminal_reason=str(
                        override.get("terminal_reason", coarse_reason)
                    ).lower(),
                    coarse_reason=coarse_reason,
                    reason_confidence=float(override.get("confidence", 1.0)),
                    reason_evidence=str(
                        override.get("reason_evidence", "manual review")
                    ),
                    needs_review=False,
                )
            )
        self.points = updated

    def _load_video_events(self, video_id, generalized):
        events = []
        for row in generalized["classes"].get("Serve", []):
            custom = row.get("custom", {})
            player = str(custom.get("Player", "")).strip().lower()
            result = str(custom.get("Result", "")).strip().lower()
            code = self._serve_class(player, result)
            if code is None:
                continue
            events.append(
                TennisEvent(
                    video_id=video_id,
                    event_id=str(row.get("name", "")),
                    kind="serve",
                    start_frame=self._frame(row.get("start")),
                    end_frame=self._frame(row.get("end")),
                    player_side=player,
                    event_class=code,
                    result=result,
                )
            )

        for row in generalized["classes"].get("Hit", []):
            custom = row.get("custom", {})
            player = str(custom.get("Player", "")).strip().lower()
            stroke_side = str(custom.get("Side", "")).strip().lower()
            code = self._hit_class(player, stroke_side)
            if code is None:
                continue
            events.append(
                TennisEvent(
                    video_id=video_id,
                    event_id=str(row.get("name", "")),
                    kind="hit",
                    start_frame=self._frame(row.get("start")),
                    end_frame=self._frame(row.get("end")),
                    player_side=player,
                    event_class=code,
                    stroke_side=stroke_side,
                    stroke_type=str(custom.get("Type", "")).strip().lower() or None,
                )
            )
        return events

    def _load_video_points(self, video_id, raw, generalized):
        raw_classes = raw["classes"]
        generalized_classes = generalized["classes"]
        raw_children = {
            category: {
                str(row.get("name")): row
                for row in raw_classes.get(category, [])
            }
            for category in ("Serve", "Hit")
        }
        generalized_children = {
            category: {
                str(row.get("name")): row
                for row in generalized_classes.get(category, [])
            }
            for category in ("Serve", "Hit")
        }
        generalized_rows = {
            category: generalized_classes.get(category, [])
            for category in ("Serve", "Hit")
        }

        points = []
        for row in raw_classes.get("Point", []):
            start = self._frame(row.get("start"))
            end = self._frame(row.get("end"))
            local_id = str(row.get("name", ""))
            custom = row.get("custom", {})
            winner_name = str(custom.get("Winner", "")).strip()
            name_to_side = self._player_side_map_from_children(
                winner_name,
                start,
                end,
                raw_children,
                generalized_children,
            )
            global_id = self._global_point_ids.get(
                (video_id, local_id),
                f"{video_id}:{local_id}",
            )
            description = str(row.get("desc", "")).strip()
            generalized_description = self._captions.get(global_id, "")
            weak_reason = infer_terminal_reason(
                generalized_description or description
            )
            winner_side = name_to_side.get(winner_name)
            point_events = [
                event
                for category in ("Serve", "Hit")
                for event in generalized_rows[category]
                if start <= self._frame(event.get("start")) <= end
            ]
            if winner_side is None:
                winner_side = self._infer_winner_side(
                    generalized_description,
                    weak_reason,
                    point_events,
                )
            points.append(
                TennisPoint(
                    video_id=video_id,
                    point_id=global_id,
                    local_id=local_id,
                    start_frame=start,
                    end_frame=end,
                    winner_side=winner_side,
                    score=str(custom.get("Score", "")).strip(),
                    description=description,
                    generalized_description=generalized_description,
                    terminal_reason=weak_reason.reason,
                    coarse_reason=weak_reason.coarse_reason,
                    reason_confidence=weak_reason.confidence,
                    reason_evidence=weak_reason.matched_text,
                    needs_review=weak_reason.needs_review or winner_side is None,
                )
            )
        return points

    @staticmethod
    def _player_side_map_from_children(
        winner_name,
        point_start,
        point_end,
        raw_children,
        generalized_children,
    ):
        name_to_side = {}
        # Serves normally cover every point; hits provide a second mapping in
        # rallies and make the conversion robust to incomplete serve labels.
        for category in ("Serve", "Hit"):
            for event_id, raw_event in raw_children[category].items():
                start = TenniSetAnnotations._frame(raw_event.get("start"))
                if not point_start <= start <= point_end:
                    continue
                generalized_event = generalized_children[category].get(event_id)
                if generalized_event is None:
                    continue
                raw_name = str(
                    raw_event.get("custom", {}).get("Player", "")
                ).strip()
                side = str(
                    generalized_event.get("custom", {}).get("Player", "")
                ).strip().lower()
                if raw_name and side in ("near", "far"):
                    name_to_side[raw_name] = side
        if winner_name and winner_name not in name_to_side and len(name_to_side) == 1:
            only_name, only_side = next(iter(name_to_side.items()))
            if winner_name != only_name:
                name_to_side[winner_name] = TenniSetAnnotations._opposite_side(
                    only_side
                )
        return name_to_side

    @staticmethod
    def _infer_winner_side(description, weak_reason, point_events):
        normalized = " ".join(str(description or "").lower().split())
        terminal_actor = None
        if normalized and weak_reason.matched_text:
            evidence_end = normalized.rfind(weak_reason.matched_text)
            evidence_end = (
                len(normalized)
                if evidence_end < 0
                else evidence_end + len(weak_reason.matched_text)
            )
            actors = list(re.finditer(r"\b(np|fp)\b", normalized[:evidence_end]))
            if actors:
                terminal_actor = {"np": "near", "fp": "far"}[
                    actors[-1].group(1)
                ]

        sorted_events = sorted(
            point_events,
            key=lambda event: TenniSetAnnotations._frame(event.get("start")),
        )
        serve_side = next(
            (
                str(event.get("custom", {}).get("Player", "")).lower()
                for event in sorted_events
                if "Result" in event.get("custom", {})
            ),
            None,
        )
        last_hit_side = next(
            (
                str(event.get("custom", {}).get("Player", "")).lower()
                for event in reversed(sorted_events)
                if "Side" in event.get("custom", {})
            ),
            None,
        )
        if serve_side not in ("near", "far"):
            serve_side = None
        if last_hit_side not in ("near", "far"):
            last_hit_side = None

        if weak_reason.reason in ("net", "out", "double_fault"):
            loser = terminal_actor or last_hit_side or serve_side
            return TenniSetAnnotations._opposite_side(loser)
        if weak_reason.reason == "unreturned":
            if terminal_actor is not None:
                return TenniSetAnnotations._opposite_side(terminal_actor)
            return serve_side
        if weak_reason.reason in ("winner", "ace"):
            return terminal_actor or last_hit_side or serve_side
        return None

    @staticmethod
    def _opposite_side(side):
        return {"near": "far", "far": "near"}.get(side)

    def _read_global_point_ids(self):
        path = self.annotation_dir / "points.txt"
        if not path.is_file():
            return {}
        counters = {}
        mapping = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            columns = line.split("\t")
            if len(columns) < 5:
                continue
            point_id, video_id = columns[0].strip(), columns[1].strip()
            local_id = columns[4].strip()
            if not point_id or not video_id or not local_id:
                continue
            # The final column is the local point name in the bundled files.
            mapping[(video_id, local_id)] = point_id
            counters[video_id] = counters.get(video_id, 0) + 1
        return mapping

    def _read_captions(self):
        path = self.annotation_dir / "captions.txt"
        if not path.is_file():
            return {}
        captions = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            point_id, separator, description = line.partition("\t")
            if separator and point_id.strip():
                captions[point_id.strip()] = description.strip()
        return captions

    def event_rows(self, video_ids: Iterable[str] | None = None):
        allowed = set(video_ids or self.video_ids)
        return [row for row in self.events if row.video_id in allowed]

    def point_rows(self, video_ids: Iterable[str] | None = None):
        allowed = set(video_ids or self.video_ids)
        return [row for row in self.points if row.video_id in allowed]

    def export(self, output_dir):
        """Write compact JSONL manifests for training and manual review."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        event_path = output_dir / "events.jsonl"
        point_path = output_dir / "points.jsonl"
        review_path = output_dir / "reason_review.jsonl"
        summary_path = output_dir / "summary.json"

        self._write_jsonl(event_path, (row.to_dict() for row in self.events))
        self._write_jsonl(point_path, (row.to_dict() for row in self.points))
        self._write_jsonl(
            review_path,
            (row.to_dict() for row in self.points if row.needs_review),
        )
        summary = {
            "videos": self.video_ids,
            "event_count": len(self.events),
            "serve_count": sum(row.kind == "serve" for row in self.events),
            "hit_count": sum(row.kind == "hit" for row in self.events),
            "point_count": len(self.points),
            "winner_labeled_points": sum(
                row.winner_side in ("near", "far") for row in self.points
            ),
            "reason_labeled_points": sum(
                row.coarse_reason in REASON_CLASSES for row in self.points
            ),
            "review_points": sum(row.needs_review for row in self.points),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "events": event_path,
            "points": point_path,
            "review": review_path,
            "summary": summary_path,
        }

    @staticmethod
    def _write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_json(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _frame(value):
        return int(round(float(value)))

    @staticmethod
    def _serve_class(player, result):
        side = {"far": "F", "near": "N"}.get(player)
        result_code = {"in": "I", "fault": "F", "let": "L"}.get(result)
        code = f"S{side}{result_code}" if side and result_code else None
        return code if code in EVENT_CLASSES else None

    @staticmethod
    def _hit_class(player, stroke_side):
        player_code = {"far": "F", "near": "N"}.get(player)
        side_code = {"left": "L", "right": "R"}.get(stroke_side)
        code = f"H{player_code}{side_code}" if player_code and side_code else None
        return code if code in EVENT_CLASSES else None
