from __future__ import annotations

from dataclasses import dataclass

from .domains.video_action_set.engine import VideoActionSetEngine
from .domains.video_action_set.episode_plan_service import EpisodePlanService
from .domains.video_action_set.facade import VideoActionSetFacade
from .domains.video_action_set.identity_service import IdentityService
from .domains.video_action_set.store import VideoActionSetStore
from .stores.video_action_set_memory import InMemoryVideoActionSetStore


@dataclass(frozen=True, slots=True)
class ZeroModelRuntime:
    video_action_set: VideoActionSetFacade


def build_runtime(
    *,
    video_action_set_store: VideoActionSetStore | None = None,
) -> ZeroModelRuntime:
    store = video_action_set_store or InMemoryVideoActionSetStore()
    identity_service = IdentityService(store=store)
    episode_plan_service = EpisodePlanService(store=store)
    engine = VideoActionSetEngine(
        identity_service=identity_service,
        episode_plan_service=episode_plan_service,
    )
    facade = VideoActionSetFacade(engine=engine)
    return ZeroModelRuntime(video_action_set=facade)


__all__ = ["ZeroModelRuntime", "build_runtime"]
