from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import NotFoundError
from app.db.models import Evaluation, Submission
from app.schemas.submissions import SubmissionCreate, SubmissionRead, SubmissionSummary
from app.services.evaluation_service import EVALUATION_LOAD_OPTIONS, to_evaluation_read

PROMPT_PREVIEW_LENGTH = 120
LIKE_ESCAPE = "\\"


def create_submission(db: Session, payload: SubmissionCreate) -> Submission:
    submission = Submission(**payload.model_dump())
    db.add(submission)
    db.commit()
    return submission


def list_submissions(
    db: Session, *, limit: int, offset: int, model_name: str | None = None
) -> tuple[list[SubmissionSummary], int]:
    """Newest submissions first, each with its number of evaluations.

    `model_name` filters by case-insensitive substring match; blank values are ignored.
    """
    model_filter = _model_name_condition(model_name)
    count_stmt = select(func.count()).select_from(Submission)
    evaluation_count = (
        select(func.count(Evaluation.id))
        .where(Evaluation.submission_id == Submission.id)
        .correlate(Submission)
        .scalar_subquery()
    )
    stmt = select(Submission, evaluation_count)
    if model_filter is not None:
        count_stmt = count_stmt.where(model_filter)
        stmt = stmt.where(model_filter)

    total = db.scalar(count_stmt) or 0
    rows = db.execute(stmt.order_by(Submission.id.desc()).limit(limit).offset(offset)).all()
    return [_to_summary(submission, count) for submission, count in rows], total


def _model_name_condition(model_name: str | None) -> ColumnElement[bool] | None:
    """Case-insensitive substring match on model_name, portable across SQLite and PostgreSQL.

    LIKE wildcards in the user's text are escaped so that `%` and `_` match literally.
    """
    term = (model_name or "").strip()
    if not term:
        return None
    escaped = (
        term.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", f"{LIKE_ESCAPE}%")
        .replace("_", f"{LIKE_ESCAPE}_")
    )
    return Submission.model_name.ilike(f"%{escaped}%", escape=LIKE_ESCAPE)


def get_submission(
    db: Session, submission_id: int, *, with_evaluations: bool = False
) -> Submission:
    stmt = select(Submission).where(Submission.id == submission_id)
    if with_evaluations:
        stmt = stmt.options(
            *(selectinload(Submission.evaluations).options(o) for o in EVALUATION_LOAD_OPTIONS)
        )
    submission = db.scalar(stmt)
    if submission is None:
        raise NotFoundError(
            f"Submission {submission_id} was not found.", code="SUBMISSION_NOT_FOUND"
        )
    return submission


def to_submission_read(submission: Submission) -> SubmissionRead:
    return SubmissionRead(
        id=submission.id,
        prompt=submission.prompt,
        output=submission.output,
        model_name=submission.model_name,
        model_version=submission.model_version,
        model_metadata=submission.model_metadata,
        reference_answer=submission.reference_answer,
        created_at=submission.created_at,
        evaluations=[to_evaluation_read(e) for e in submission.evaluations],
    )


def _to_summary(submission: Submission, evaluation_count: int) -> SubmissionSummary:
    prompt = submission.prompt
    if len(prompt) > PROMPT_PREVIEW_LENGTH:
        prompt = prompt[: PROMPT_PREVIEW_LENGTH - 1].rstrip() + "…"
    return SubmissionSummary(
        id=submission.id,
        model_name=submission.model_name,
        model_version=submission.model_version,
        prompt_preview=prompt,
        evaluation_count=evaluation_count,
        created_at=submission.created_at,
    )
