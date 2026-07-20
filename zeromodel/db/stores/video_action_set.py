from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from ...domains.video_action_set.dto import BenchmarkIdentityDTO
from ...domains.video_action_set.store import (
    VideoActionSetStore,
    raise_identity_conflict,
)
from ..orm.video_action_set import BenchmarkIdentityORM


class SqlAlchemyVideoActionSetStore(VideoActionSetStore):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(BenchmarkIdentityORM, identity.seed_digest)
                if existing is not None:
                    existing_dto = self._to_dto(existing)
                    if existing_dto != identity:
                        raise_identity_conflict()
                    return existing_dto
                session.add(self._to_orm(identity))
            return identity
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(BenchmarkIdentityORM, seed_digest)
                if existing is None:
                    return None
                return self._to_dto(existing)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _to_dto(identity: BenchmarkIdentityORM) -> BenchmarkIdentityDTO:
        return BenchmarkIdentityDTO(
            contract_commit=identity.contract_commit,
            seed_material=identity.seed_material,
            seed_digest=identity.seed_digest,
            policy_artifact_id=identity.policy_artifact_id,
            parent_audit_sha=identity.parent_audit_sha,
            parent_v3_sha=identity.parent_v3_sha,
        )

    @staticmethod
    def _to_orm(identity: BenchmarkIdentityDTO) -> BenchmarkIdentityORM:
        return BenchmarkIdentityORM(
            contract_commit=identity.contract_commit,
            seed_material=identity.seed_material,
            seed_digest=identity.seed_digest,
            policy_artifact_id=identity.policy_artifact_id,
            parent_audit_sha=identity.parent_audit_sha,
            parent_v3_sha=identity.parent_v3_sha,
        )


__all__ = ["SqlAlchemyVideoActionSetStore"]
