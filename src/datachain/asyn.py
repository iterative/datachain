import asyncio
from collections.abc import AsyncIterable, Awaitable, Coroutine, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from heapq import heappop, heappush
from typing import Any, Callable, Generic, Optional, TypeVar

from fsspec.asyn import get_loop

ASYNC_WORKERS = 20

InputT = TypeVar("InputT", contravariant=True)  # noqa: PLC0105
ResultT = TypeVar("ResultT", covariant=True)  # noqa: PLC0105
T = TypeVar("T")


class AsyncMapper(Generic[InputT, ResultT]):
    """
    Asynchronous unordered mapping iterable compatible with fsspec.

    `AsyncMapper(func, it)` is roughly equivalent to `map(func, it)`, except
    that `func` is an async function, which is executed concurrently by up to
    `workers` asyncio coroutines, and the results are yielded in arbitrary
    order.

    If `func` needs to call synchronous functions that may themselves run coroutines
    on the same loop, it must use `mapper.to_thread()`.

    Note that `loop`, which defaults to the fsspec loop, must be running on a different
    thread than the one calling `mapper.iterate()`.
    """

    order_preserving = False
    work_queue: "asyncio.Queue[InputT]"
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        func: Callable[[InputT], Awaitable[ResultT]],
        iterable: Iterable[InputT],
        *,
        workers: int = ASYNC_WORKERS,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.func = func
        self.iterable = iterable
        self.workers = workers
        self.loop = get_loop() if loop is None else loop
        self.pool = ThreadPoolExecutor(workers)
        self._tasks: set[asyncio.Task] = set()

    def start_task(self, coro: Coroutine) -> asyncio.Task:
        task = self.loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def produce(self) -> None:
        for item in self.iterable:
            await self.work_queue.put(item)

    async def worker(self) -> None:
        while (item := await self.work_queue.get()) is not None:
            try:
                result = await self.func(item)
                await self.result_queue.put(result)
            finally:
                self.work_queue.task_done()

    async def init(self) -> None:
        self.work_queue = asyncio.Queue(2 * self.workers)
        self.result_queue: asyncio.Queue[Optional[ResultT]] = asyncio.Queue(
            self.workers
        )

    async def run(self) -> None:
        producer = self.start_task(self.produce())
        for _i in range(self.workers):
            self.start_task(self.worker())
        try:
            done, _pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.gather_exceptions(done)
            assert producer.done()
            join = self.start_task(self.work_queue.join())
            done, _pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.gather_exceptions(done)
            assert join.done()
        except:
            await self.cancel_all()
            await self._break_iteration()
            raise
        else:
            await self.cancel_all()
            await self._end_iteration()

    async def cancel_all(self) -> None:
        if self._tasks:
            for task in self._tasks:
                task.cancel()
            await asyncio.wait(self._tasks)

    def gather_exceptions(self, done_tasks):
        # Check all exceptions to avoid "Task exception was never retrieved" warning
        exceptions = [task.exception() for task in done_tasks]
        # Raise the first exception found, if any. Additional ones are ignored.
        for exc in exceptions:
            if exc:
                raise exc

    async def _pop_result(self) -> Optional[ResultT]:
        return await self.result_queue.get()

    def next_result(self, timeout=None) -> Optional[ResultT]:
        """
        Return the next available result.

        Blocks as long as the result queue is empty.
        """
        future = asyncio.run_coroutine_threadsafe(self._pop_result(), self.loop)
        return future.result(timeout=timeout)

    async def _end_iteration(self) -> None:
        """Signal successful end of iteration."""
        await self.result_queue.put(None)

    async def _break_iteration(self) -> None:
        """Signal that iteration must stop ASAP."""
        while not self.result_queue.empty():
            self.result_queue.get_nowait()
        await self.result_queue.put(None)

    def iterate(self, timeout=None) -> Iterable[ResultT]:
        init = asyncio.run_coroutine_threadsafe(self.init(), self.loop)
        init.result(timeout=1)
        async_run = asyncio.run_coroutine_threadsafe(self.run(), self.loop)
        try:
            while True:
                if (result := self.next_result(timeout)) is not None:
                    yield result
                else:
                    break
            if exc := async_run.exception():
                raise exc
        finally:
            if not async_run.done():
                async_run.cancel()

    def __iter__(self):
        return self.iterate()

    async def to_thread(self, func, *args):
        return await self.loop.run_in_executor(self.pool, func, *args)


class OrderedMapper(AsyncMapper[InputT, ResultT]):
    """
    Asynchronous ordered mapping iterable compatible with fsspec.

    See `AsyncMapper` for details.
    """

    order_preserving = True

    def __init__(
        self,
        func: Callable[[InputT], Awaitable[ResultT]],
        iterable: Iterable[InputT],
        *,
        workers: int = ASYNC_WORKERS,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(func, iterable, workers=workers, loop=loop)
        self._waiters: dict[int, Any] = {}
        self._getters: dict[int, asyncio.Future[Optional[ResultT]]] = {}
        self.heap: list[tuple[int, Optional[ResultT]]] = []
        self._next_yield = 0
        self._items_seen = 0
        self._window = 2 * workers

    def _push_result(self, i: int, result: Optional[ResultT]) -> None:
        if i in self._getters:
            future = self._getters.pop(i)
            future.set_result(result)
        else:
            heappush(self.heap, (i, result))

    async def worker(self) -> None:
        while (item := await self.work_queue.get()) is not None:
            i = self._items_seen
            self._items_seen += 1
            if i >= self._next_yield + self._window:
                event = self._waiters[i - self._window] = asyncio.Event()
                await event.wait()
            result = await self.func(item)
            self._push_result(i, result)
            self.work_queue.task_done()

    async def init(self) -> None:
        self.work_queue = asyncio.Queue(2 * self.workers)

    async def _pop_result(self) -> Optional[ResultT]:
        if self.heap and self.heap[0][0] == self._next_yield:
            _i, out = heappop(self.heap)
        else:
            self._getters[self._next_yield] = get_value = self.loop.create_future()
            out = await get_value
        if self._next_yield in self._waiters:
            event = self._waiters.pop(self._next_yield)
            event.set()
        self._next_yield += 1
        return out

    async def _end_iteration(self) -> None:
        self._push_result(self._next_yield + len(self.heap), None)

    async def _break_iteration(self) -> None:
        self.heap = []
        self._push_result(self._next_yield, None)


def iter_over_async(ait: AsyncIterable[T], loop) -> Iterator[T]:
    """Wrap an asynchronous iterator into a synchronous one"""
    ait = ait.__aiter__()

    # helper async fn that just gets the next element from the async iterator
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    # actual sync iterator
    while True:
        done, obj = asyncio.run_coroutine_threadsafe(get_next(), loop).result()
        if done:
            break
        yield obj
