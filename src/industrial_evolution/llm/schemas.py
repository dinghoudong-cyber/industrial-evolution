from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["no_update", "retrain_with_replay"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)
