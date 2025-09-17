import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Union,
)


@dataclass
class Checkpoint:
    """
    Class that represents checkpoint in job run. Checkpoint means that job has
    successfully ran until that point and in a case of a failure, it can continue
    from that.
    Checkpoint has also a special "mode" called partial which means that it's not
    completely done, e.g in half way of running UDF something fails but we save
    already calculated results and continue with it on restart.
    """

    id: str
    job_id: str
    hash: str
    partial: bool
    created_at: datetime

    @classmethod
    def parse(
        cls,
        id: Union[str, uuid.UUID],
        job_id: str,
        _hash: str,
        partial: bool,
        created_at: datetime,
    ) -> "Checkpoint":
        return cls(
            str(id),
            job_id,
            _hash,
            bool(partial),
            created_at,
        )
