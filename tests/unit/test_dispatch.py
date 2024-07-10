from queue import Empty, Full
from typing import Optional

from datachain.query.dispatch import (
    STOP_SIGNAL,
    UDFDispatcher,
    get_from_queue,
    put_into_queue,
)


class MockQueue:
    def __init__(self) -> None:
        self.return_empty_once = True
        self.return_full_once = True
        self.put_signal: Optional[str] = None
        self.put_count: int = 0

    def put_nowait(self, task: str) -> None:
        if self.return_full_once:
            self.return_full_once = False
            raise Full
        assert task == STOP_SIGNAL
        self.put_signal = task
        self.put_count += 1

    def get_nowait(self) -> str:
        if self.return_empty_once:
            self.return_empty_once = False
            raise Empty
        return STOP_SIGNAL


def test_get_from_queue():
    mock_queue = MockQueue()

    assert get_from_queue(mock_queue) == STOP_SIGNAL


def test_put_into_queue():
    mock_queue = MockQueue()

    assert put_into_queue(mock_queue, STOP_SIGNAL) is None
    assert mock_queue.put_signal == STOP_SIGNAL


def test_send_stop_signal_to_workers():
    mock_queue = MockQueue()

    UDFDispatcher.send_stop_signal_to_workers(mock_queue, 8)

    assert mock_queue.put_signal == STOP_SIGNAL
    assert mock_queue.put_count == 8
