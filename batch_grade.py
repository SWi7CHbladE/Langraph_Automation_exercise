#!/usr/bin/env python3
"""
RubricGraph: Batch handwritten-answer grader.

Takes a folder of handwritten answer images and produces one .txt and one .json
file per image.

Pipeline:

    image
      -> Ollama Vision OCR
      -> normalized plaintext
      -> LangGraph rubric grading
      -> exact rubric/evidence mapping
      -> deterministic score
      -> independent verification
      -> TXT + JSON

Example:

    python batch_grade.py \
        --input-dir handwritten_answers \
        --output-dir graded_answers \
        --rules grading_rules.txt \
        --question-file question.txt \
        --reference-file reference_answer.txt

Required Ollama models:

    ollama serve
    ollama pull qwen2.5vl:7b
    ollama pull qwen3:8b

Environment variables:

    RUBRICGRAPH_MODEL       Text/grading model (default: qwen3:8b)
    RUBRICGRAPH_OCR_MODEL   Vision/OCR model (default: qwen2.5vl:7b)
    OLLAMA_BASE_URL         Ollama URL (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = os.getenv("RUBRICGRAPH_MODEL", "qwen3:8b")
DEFAULT_OCR_MODEL = os.getenv("RUBRICGRAPH_OCR_MODEL", "qwen2.5vl:7b")
DEFAULT_OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Criterion(BaseModel):
    id: int
    name: str
    requirements: list[str]
    max_marks: float
    partial_credit_allowed: bool = True


class Rubric(BaseModel):
    question: str
    maximum_marks: float
    criteria: list[Criterion]
    global_rules: list[str]


class Claim(BaseModel):
    text: str
    concept: str
    assessment: Literal[
        "correct",
        "incorrect",
        "partial",
        "uncertain",
    ]


class AnswerAnalysis(BaseModel):
    claims: list[Claim]
    key_concepts_present: list[str]
    key_concepts_missing: list[str]


class EvidenceMapping(BaseModel):
    criterion_id: int
    criterion_name: str

    # These must be copied from the student's plaintext answer.
    supporting_answer_parts: list[str]

    # Which rubric requirements those exact answer parts represent.
    represented_requirements: list[str]

    evidence_status: Literal[
        "sufficient",
        "partial",
        "incorrect",
        "absent",
        "uncertain",
    ]

    explanation: str


class CriterionResult(BaseModel):
    criterion_id: int
    criterion_name: str
    awarded_marks: float
    maximum_marks: float
    evidence: EvidenceMapping
    missing_requirements: list[str]
    reasoning: str


class VerificationResult(BaseModel):
    valid: bool
    issues: list[str]
    requires_regrading: bool


class FinalReport(BaseModel):
    total_score: float
    maximum_score: float
    percentage: float
    criterion_results: list[CriterionResult]
    overall_feedback: str


class OCRResult(BaseModel):
    raw_transcription: str
    plaintext_answer: str
    uncertain_segments: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

def append_results(existing, new):
    existing = existing or []

    if new is None:
        return existing

    if isinstance(new, CriterionResult):
        return existing + [new]

    return existing + list(new)


class GradingState(TypedDict, total=False):
    rules_path: str
    rules_text: str
    rubric: Rubric

    question: str
    reference_answer: str
    student_answer: str

    answer_analysis: AnswerAnalysis

    criterion_results: Annotated[
        list[CriterionResult],
        append_results,
    ]

    total_score: float
    verification: VerificationResult
    regrade_count: int
    final_report: FinalReport


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"File is empty: {path}")

    return text


def image_to_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path.name)

    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(
            f"Could not determine a supported image MIME type for "
            f"{image_path}"
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_handwritten_image(
    image_path: Path,
    ocr_llm: ChatOllama,
) -> OCRResult:
    image_uri = image_to_data_uri(image_path)

    structured_ocr = ocr_llm.with_structured_output(OCRResult)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
Transcribe the handwritten answer in this image.

Return TWO versions.

1. raw_transcription:
   - Preserve the visible wording as closely as possible.
   - Preserve the student's mistakes.
   - Do not correct factual or technical statements.
   - Do not improve grammar.
   - Do not infer missing text.
   - Use [ILLEGIBLE] for genuinely unreadable text.

2. plaintext_answer:
   - Convert the transcription into clean plain text.
   - Preserve the student's wording and technical meaning.
   - Normalize whitespace and line breaks.
   - Do NOT rewrite an incorrect answer into a correct answer.
   - Do NOT add information that is not visible.
   - Do NOT remove meaningful student statements.

Also return:
- uncertain_segments: genuinely ambiguous portions
- confidence: conservative OCR confidence from 0 to 1

The plaintext answer will be used as evidence for grading.
Fidelity is more important than fluency.
""",
            },
            {
                "type": "image_url",
                "image_url": {"url": image_uri},
            },
        ],
    )

    return structured_ocr.invoke([message])


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def load_rules(state: GradingState):
    rules = read_text_file(Path(state["rules_path"]))
    return {"rules_text": rules}


def parse_rubric(state: GradingState):
    parser = llm.with_structured_output(Rubric)

    prompt = f"""
Convert this plaintext grading policy into the Rubric schema.

The plaintext policy is authoritative.

Requirements:
- Preserve every criterion.
- Preserve maximum marks.
- Preserve every criterion requirement.
- Preserve all global grading rules.
- Do not invent criteria.

PLAINTEXT GRADING POLICY:
-------------------------
{state["rules_text"]}
-------------------------
"""

    rubric = parser.invoke(prompt)
    return {"rubric": rubric}


def analyze_answer(state: GradingState):
    analyzer = llm.with_structured_output(AnswerAnalysis)

    prompt = f"""
Analyze the student's answer without assigning a score.

Question:
{state["question"]}

Reference answer:
{state["reference_answer"]}

Student plaintext answer:
-------------------------
{state["student_answer"]}
-------------------------

Extract:
1. explicit claims made by the student,
2. the concept represented by each claim,
3. whether each claim is correct, incorrect, partial, or uncertain,
4. important concepts present,
5. important concepts missing.

Do not infer knowledge that is not expressed.
Do not correct the student's answer.
"""

    analysis = analyzer.invoke(prompt)
    return {"answer_analysis": analysis}


def fan_out_criteria(state: GradingState):
    return [
        Send(
            "evaluate_criterion",
            {
                "rubric": state["rubric"],
                "question": state["question"],
                "reference_answer": state["reference_answer"],
                "student_answer": state["student_answer"],
                "answer_analysis": state["answer_analysis"],
                "criterion": criterion,
            },
        )
        for criterion in state["rubric"].criteria
    ]


def evaluate_criterion(state: dict):
    criterion: Criterion = state["criterion"]

    evaluator = llm.with_structured_output(CriterionResult)

    prompt = f"""
You are grading ONE criterion of a student's answer.

Create an auditable mapping:

RUBRIC REQUIREMENT
        ->
EXACT STUDENT ANSWER PART
        ->
AWARDED MARKS

Question:
{state["question"]}

Reference answer:
{state["reference_answer"]}

Student plaintext answer:
-------------------------
{state["student_answer"]}
-------------------------

Answer analysis:
{state["answer_analysis"].model_dump_json(indent=2)}

Criterion:
{criterion.model_dump_json(indent=2)}

Global grading rules:
{state["rubric"].global_rules}

Instructions:

1. Examine the student's answer literally.
2. Identify the exact sentence(s), clause(s), phrase(s), equation(s),
   or other part(s) that represent this criterion.
3. Copy those answer parts VERBATIM into supporting_answer_parts.
4. Do not paraphrase the student's evidence.
5. State which rubric requirement each selected answer part represents.
6. If no answer part addresses the criterion, use [] and evidence_status
   "absent".
7. If the relevant evidence is factually wrong, use "incorrect".
8. If only part of the criterion is satisfied, use "partial".
9. Do not award marks merely because a keyword appears.
10. Do not infer knowledge that the student did not express.
11. Do not rewrite incorrect student statements into correct statements.
12. Do not require wording identical to the reference answer.
13. Do not penalize grammar unless technical meaning is unclear.
14. Never exceed {criterion.max_marks} marks.
15. Keep missing_requirements explicit.
16. Every supporting_answer_parts entry must actually occur in the
    supplied student plaintext answer.

Return a complete CriterionResult.
"""

    result = evaluator.invoke(prompt)

    result.awarded_marks = max(
        0.0,
        min(float(result.awarded_marks), float(criterion.max_marks)),
    )
    result.maximum_marks = float(criterion.max_marks)

    # Metadata is controlled by application state, not by the LLM.
    result.criterion_id = criterion.id
    result.criterion_name = criterion.name
    result.evidence.criterion_id = criterion.id
    result.evidence.criterion_name = criterion.name

    return {"criterion_results": [result]}


def aggregate_score(state: GradingState):
    results = state.get("criterion_results", [])

    total = sum(
        float(result.awarded_marks)
        for result in results
    )

    maximum = float(state["rubric"].maximum_marks)

    # Defensive bound.
    total = max(0.0, min(total, maximum))

    return {"total_score": total}


def verify_grade(state: GradingState):
    verifier = llm.with_structured_output(VerificationResult)

    results = sorted(
        state.get("criterion_results", []),
        key=lambda r: r.criterion_id,
    )

    prompt = f"""
Verify this proposed grading decision.

Question:
{state["question"]}

Student plaintext answer:
{state["student_answer"]}

Rubric:
{state["rubric"].model_dump_json(indent=2)}

Criterion results:
{json.dumps(
    [r.model_dump() for r in results],
    indent=2,
    ensure_ascii=False,
)}

Deterministically calculated total:
{state["total_score"]}

Check every criterion:

1. Does each cited evidence excerpt actually occur in the student answer?
2. Was the evidence copied faithfully rather than invented?
3. Does each cited answer part address the stated rubric requirement?
4. Is the evidence factually correct?
5. Are partial marks justified?
6. Are missing requirements correctly identified?
7. Was unsupported knowledge inferred?
8. Does any criterion exceed its maximum?
9. Is the total mathematically consistent?
10. Do all decisions follow the global grading rules?

If a substantive grading problem exists, set requires_regrading=true.

Do not assign a new score. Only report verification issues.
"""

    verification = verifier.invoke(prompt)
    return {"verification": verification}


def route_after_verification(state: GradingState):
    if (
        state["verification"].requires_regrading
        and state.get("regrade_count", 0) < 1
    ):
        return "regrade"

    return "final_report"


def regrade(state: GradingState):
    evaluator = llm.with_structured_output(CriterionResult)

    issues = state["verification"].issues
    corrected = []

    for criterion in state["rubric"].criteria:
        prompt = f"""
Re-evaluate this criterion after independent verification.

Criterion:
{criterion.model_dump_json(indent=2)}

Question:
{state["question"]}

Student plaintext answer:
-------------------------
{state["student_answer"]}
-------------------------

Previous results:
{json.dumps(
    [r.model_dump() for r in state.get("criterion_results", [])],
    indent=2,
    ensure_ascii=False,
)}

Verifier issues:
{json.dumps(issues, indent=2, ensure_ascii=False)}

Global rules:
{state["rubric"].global_rules}

Rules:
- Correct only substantive problems exposed by verification.
- Supporting answer parts MUST be copied from the student's answer.
- Never invent evidence.
- Never rewrite student statements into correct statements.
- Do not infer unstated knowledge.
- Do not exceed the criterion maximum.
"""

        result = evaluator.invoke(prompt)

        result.awarded_marks = max(
            0.0,
            min(float(result.awarded_marks), float(criterion.max_marks)),
        )
        result.maximum_marks = float(criterion.max_marks)
        result.criterion_id = criterion.id
        result.criterion_name = criterion.name
        result.evidence.criterion_id = criterion.id
        result.evidence.criterion_name = criterion.name

        corrected.append(result)

    return {
        "criterion_results": corrected,
        "regrade_count": state.get("regrade_count", 0) + 1,
    }


def generate_overall_feedback(state: GradingState) -> str:
    class Feedback(BaseModel):
        overall_feedback: str

    feedback_llm = llm.with_structured_output(Feedback)

    results = sorted(
        state.get("criterion_results", []),
        key=lambda r: r.criterion_id,
    )

    prompt = f"""
Write concise technical feedback for the student.

Do not change any marks.
Do not introduce information that is absent from the criterion results.

Criterion results:
{json.dumps(
    [r.model_dump() for r in results],
    indent=2,
    ensure_ascii=False,
)}
"""

    return feedback_llm.invoke(prompt).overall_feedback


def final_report(state: GradingState):
    results = sorted(
        state.get("criterion_results", []),
        key=lambda r: r.criterion_id,
    )

    maximum = float(state["rubric"].maximum_marks)
    total = float(state["total_score"])

    percentage = (
        100.0 * total / maximum
        if maximum > 0
        else 0.0
    )

    report = FinalReport(
        total_score=total,
        maximum_score=maximum,
        percentage=percentage,
        criterion_results=results,
        overall_feedback=generate_overall_feedback(state),
    )

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(GradingState)

    builder.add_node("load_rules", load_rules)
    builder.add_node("parse_rubric", parse_rubric)
    builder.add_node("analyze_answer", analyze_answer)
    builder.add_node("evaluate_criterion", evaluate_criterion)
    builder.add_node("aggregate_score", aggregate_score)
    builder.add_node("verify_grade", verify_grade)
    builder.add_node("regrade", regrade)
    builder.add_node("final_report", final_report)

    builder.add_edge(START, "load_rules")
    builder.add_edge("load_rules", "parse_rubric")
    builder.add_edge("parse_rubric", "analyze_answer")

    # Dynamic fan-out: one branch for each marking criterion.
    builder.add_conditional_edges(
        "analyze_answer",
        fan_out_criteria,
        ["evaluate_criterion"],
    )

    # Criterion branches converge.
    builder.add_edge(
        "evaluate_criterion",
        "aggregate_score",
    )

    builder.add_edge(
        "aggregate_score",
        "verify_grade",
    )

    builder.add_conditional_edges(
        "verify_grade",
        route_after_verification,
        {
            "regrade": "regrade",
            "final_report": "final_report",
        },
    )

    builder.add_edge(
        "regrade",
        "aggregate_score",
    )

    builder.add_edge(
        "final_report",
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def build_text_report(
    result: GradingState,
    source_image: Path,
    ocr_result: OCRResult,
) -> str:
    report = result["final_report"]
    rubric = result["rubric"]
    verification = result["verification"]

    lines = [
        "=" * 100,
        "RUBRICGRAPH GRADING REPORT",
        "=" * 100,
        f"Source image: {source_image}",
        f"Question: {result['question']}",
        f"Final score: {report.total_score:.1f}/{report.maximum_score:.1f} "
        f"({report.percentage:.1f}%)",
        "",
        "OCR",
        "-" * 100,
        f"Confidence: {ocr_result.confidence:.3f}",
        f"Uncertain segments: {json.dumps(ocr_result.uncertain_segments, ensure_ascii=False)}",
        "",
        "RAW OCR TRANSCRIPTION",
        "-" * 100,
        ocr_result.raw_transcription,
        "",
        "NORMALIZED PLAINTEXT ANSWER",
        "-" * 100,
        result["student_answer"],
        "",
        "RUBRIC-BY-RUBRIC EVIDENCE",
        "=" * 100,
    ]

    for criterion in rubric.criteria:
        r = next(
            x for x in report.criterion_results
            if x.criterion_id == criterion.id
        )

        lines.extend([
            "",
            f"RUBRIC {criterion.id}: {criterion.name}",
            f"MARKS: {r.awarded_marks:.1f}/{r.maximum_marks:.1f}",
            f"EVIDENCE STATUS: {r.evidence.evidence_status.upper()}",
            "",
            "RUBRIC REQUIREMENTS:",
        ])

        for requirement in criterion.requirements:
            lines.append(f"  - {requirement}")

        lines.extend([
            "",
            "STUDENT ANSWER SENTENCE(S)/PART(S) REPRESENTING THIS RUBRIC:",
        ])

        if r.evidence.supporting_answer_parts:
            for part in r.evidence.supporting_answer_parts:
                lines.append(f'  "{part}"')
        else:
            lines.append("  [NONE]")

        lines.extend([
            "",
            "REPRESENTED REQUIREMENTS:",
        ])

        if r.evidence.represented_requirements:
            for req in r.evidence.represented_requirements:
                lines.append(f"  - {req}")
        else:
            lines.append("  [NONE]")

        lines.extend([
            "",
            "MISSING REQUIREMENTS:",
        ])

        if r.missing_requirements:
            for req in r.missing_requirements:
                lines.append(f"  - {req}")
        else:
            lines.append("  [NONE]")

        lines.extend([
            "",
            "EVIDENCE EXPLANATION:",
            r.evidence.explanation,
            "",
            "GRADING REASONING:",
            r.reasoning,
            "-" * 100,
        ])

    lines.extend([
        "",
        "VERIFICATION",
        "=" * 100,
        f"Valid: {verification.valid}",
        f"Regrading performed: {result.get('regrade_count', 0) > 0}",
    ])

    if verification.issues:
        lines.append("")
        lines.append("Verifier issues:")
        for issue in verification.issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("No verifier issues reported.")

    lines.extend([
        "",
        "OVERALL FEEDBACK",
        "-" * 100,
        report.overall_feedback,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-image grading
# ---------------------------------------------------------------------------

def grade_one_image(
    image_path: Path,
    output_dir: Path,
    rules_path: Path,
    question: str,
    reference_answer: str,
    graph,
    ocr_llm: ChatOllama,
    min_ocr_confidence: float,
) -> dict:
    print(f"[OCR]    {image_path.name}")

    ocr_result = ocr_handwritten_image(
        image_path,
        ocr_llm,
    )

    # Always preserve OCR output.
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    if ocr_result.confidence < min_ocr_confidence:
        text = "\n".join([
            "RUBRICGRAPH - HUMAN REVIEW REQUIRED",
            "=" * 80,
            f"Source image: {image_path}",
            f"OCR confidence: {ocr_result.confidence:.3f}",
            f"Required confidence: {min_ocr_confidence:.3f}",
            "",
            "RAW OCR TRANSCRIPTION",
            "-" * 80,
            ocr_result.raw_transcription,
            "",
            "NORMALIZED PLAINTEXT ANSWER",
            "-" * 80,
            ocr_result.plaintext_answer,
            "",
            "UNCERTAIN SEGMENTS",
            "-" * 80,
            *[
                f"- {segment}"
                for segment in ocr_result.uncertain_segments
            ],
            "",
            "STATUS: HUMAN REVIEW REQUIRED",
        ])

        txt_path.write_text(text, encoding="utf-8")

        write_json(
            json_path,
            {
                "status": "human_review_required",
                "source_image": str(image_path),
                "ocr": ocr_result.model_dump(),
                "reason": (
                    f"OCR confidence {ocr_result.confidence:.3f} is below "
                    f"the configured threshold "
                    f"{min_ocr_confidence:.3f}."
                ),
            },
        )

        print(f"[REVIEW] {image_path.name}")
        return {
            "status": "human_review_required",
            "txt": str(txt_path),
            "json": str(json_path),
        }

    print(f"[GRADE]  {image_path.name}")

    graph_result = graph.invoke({
        "rules_path": str(rules_path),
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": ocr_result.plaintext_answer,
        "regrade_count": 0,
    })

    text_report = build_text_report(
        graph_result,
        image_path,
        ocr_result,
    )

    txt_path.write_text(
        text_report,
        encoding="utf-8",
    )

    final_report = graph_result["final_report"]

    json_report = {
        "status": "graded",
        "source_image": str(image_path),
        "question": question,
        "reference_answer": reference_answer,
        "ocr": ocr_result.model_dump(),
        "rubric": graph_result["rubric"].model_dump(),
        "answer_analysis": graph_result["answer_analysis"].model_dump(),
        "criterion_results": [
            r.model_dump()
            for r in final_report.criterion_results
        ],
        "total_score": final_report.total_score,
        "maximum_score": final_report.maximum_score,
        "percentage": final_report.percentage,
        "verification": graph_result["verification"].model_dump(),
        "regrade_count": graph_result.get("regrade_count", 0),
        "overall_feedback": final_report.overall_feedback,
    }

    write_json(json_path, json_report)

    print(
        f"[DONE]   {image_path.name}: "
        f"{final_report.total_score:.1f}/"
        f"{final_report.maximum_score:.1f}"
    )

    return {
        "status": "graded",
        "score": final_report.total_score,
        "maximum": final_report.maximum_score,
        "txt": str(txt_path),
        "json": str(json_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-grade handwritten answers using Ollama Vision + "
            "LangGraph."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing handwritten answer images.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where TXT and JSON reports will be written.",
    )

    parser.add_argument(
        "--rules",
        required=True,
        type=Path,
        help="Plaintext grading-rules file.",
    )

    parser.add_argument(
        "--question-file",
        required=True,
        type=Path,
        help="Plaintext file containing the question.",
    )

    parser.add_argument(
        "--reference-file",
        required=True,
        type=Path,
        help="Plaintext file containing the reference answer.",
    )

    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL,
        help=f"Local Ollama grading model. Default: {DEFAULT_TEXT_MODEL}",
    )

    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help=f"Local Ollama vision model. Default: {DEFAULT_OCR_MODEL}",
    )

    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL. Default: {DEFAULT_OLLAMA_URL}",
    )

    parser.add_argument(
        "--ocr-threshold",
        type=float,
        default=0.80,
        help="Minimum OCR confidence required for automatic grading.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_dir}"
        )

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {args.input_dir}"
        )

    question = read_text_file(args.question_file)
    reference_answer = read_text_file(args.reference_file)

    if not (0.0 <= args.ocr_threshold <= 1.0):
        raise ValueError("--ocr-threshold must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(
            f"No supported images found in {args.input_dir}",
            file=sys.stderr,
        )
        return 1

    print("=" * 80)
    print("RUBRICGRAPH BATCH GRADER")
    print("=" * 80)
    print(f"Images:       {len(image_paths)}")
    print(f"Input:        {args.input_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Rules:        {args.rules}")
    print(f"Text model:   {args.text_model}")
    print(f"OCR model:    {args.ocr_model}")
    print(f"Ollama URL:   {args.ollama_url}")
    print(f"OCR threshold:{args.ocr_threshold:.2f}")
    print()

    global llm

    llm = ChatOllama(
        model=args.text_model,
        temperature=0,
        base_url=args.ollama_url,
    )

    ocr_llm = ChatOllama(
        model=args.ocr_model,
        temperature=0,
        base_url=args.ollama_url,
    )

    print("[INIT] Building LangGraph...")
    graph = build_graph()
    print("[INIT] Graph ready.")
    print()

    results = []

    for image_path in image_paths:
        try:
            result = grade_one_image(
                image_path=image_path,
                output_dir=args.output_dir,
                rules_path=args.rules,
                question=question,
                reference_answer=reference_answer,
                graph=graph,
                ocr_llm=ocr_llm,
                min_ocr_confidence=args.ocr_threshold,
            )
            results.append(
                {
                    "image": image_path.name,
                    **result,
                }
            )

        except Exception as exc:
            # Do not stop an entire batch because one image failed.
            print(
                f"[ERROR]  {image_path.name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            results.append(
                {
                    "image": image_path.name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "rules_file": str(args.rules),
        "question_file": str(args.question_file),
        "reference_file": str(args.reference_file),
        "text_model": args.text_model,
        "ocr_model": args.ocr_model,
        "ollama_url": args.ollama_url,
        "ocr_threshold": args.ocr_threshold,
        "image_count": len(image_paths),
        "graded_count": sum(
            r.get("status") == "graded"
            for r in results
        ),
        "human_review_count": sum(
            r.get("status") == "human_review_required"
            for r in results
        ),
        "error_count": sum(
            r.get("status") == "error"
            for r in results
        ),
        "results": results,
    }

    write_json(
        args.output_dir / "_batch_summary.json",
        summary,
    )

    print()
    print("=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Graded:         {summary['graded_count']}")
    print(f"Human review:   {summary['human_review_count']}")
    print(f"Errors:         {summary['error_count']}")
    print(f"Summary:        {args.output_dir / '_batch_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
