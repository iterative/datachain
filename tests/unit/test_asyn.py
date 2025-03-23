import asyncio
import functools
import itertools
import threading
from collections import Counter
from contextlib import contextmanager
from queue import Queue

import pytest
from fsspec.asyn import sync
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from datachain.asyn import AsyncMapper, OrderedMapper, get_loop, iter_over_async


async def fake_io(i):
    # print(f"task {i}")
    await asyncio.sleep(i)
    return i


def join_all_tasks(loop, timeout=4):
    tasks = asyncio.all_tasks(loop)
    if tasks:
        coro = asyncio.wait(tasks, timeout=timeout)
        future = asyncio.run_coroutine_threadsafe(coro, loop=loop)
        try:
            # Time out if any tasks are still running
            future.result(timeout=timeout)
        finally:
            # Avoid annoying warning after a timeout
            for t in asyncio.all_tasks(loop):
                t.cancel()


@contextmanager
def mock_time(loop):
    from aiotools import VirtualClock

    clock = VirtualClock()
    cm = clock.patch_loop()

    async def patch():
        cm.__enter__()

    async def unpatch():
        cm.__exit__(None, None, None)

    asyncio.run_coroutine_threadsafe(patch(), loop).result()
    try:
        yield
    finally:
        asyncio.run_coroutine_threadsafe(unpatch(), loop).result()


@pytest.fixture
def loop():
    loop = get_loop()
    with mock_time(loop):
        try:
            yield loop
        finally:
            join_all_tasks(loop)


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
def test_mapper_fsspec(create_mapper, loop):
    n_rows = 50

    async def process(row):
        await mapper.to_thread(functools.partial(sync, loop, fake_io, row))
        return row

    mapper = create_mapper(process, range(n_rows), workers=10, loop=loop)
    result = [sync(loop, fake_io, i + n_rows) for i in mapper.iterate(timeout=4)]
    if mapper.order_preserving:
        assert result == list(range(n_rows, 2 * n_rows))
    else:
        assert set(result) == set(range(n_rows, 2 * n_rows))


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
def test_mapper_generator_shutdown(create_mapper, loop):
    """
    Check that throwing an exception into AsyncMapper.iterate() terminates it cleanly.
    Note that finalising a generator involves throwing StopIteration into it.
    """

    async def process(row):
        await mapper.to_thread(functools.partial(sync, loop, fake_io, row))
        return row

    class MyError(Exception):
        pass

    mapper = create_mapper(process, range(50), workers=10, loop=loop)
    iterator = mapper.iterate(timeout=4)
    next(iterator)
    with pytest.raises(MyError):
        iterator.throw(MyError)


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
def test_mapper_exception_while_processing(create_mapper, loop):
    async def process(row):
        await mapper.to_thread(functools.partial(sync, loop, fake_io, row))
        if row == 12:
            raise RuntimeError
        return row

    mapper = create_mapper(process, range(50), workers=10, loop=loop)
    with pytest.raises(RuntimeError):
        list(mapper.iterate(timeout=4))


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
def test_mapper_deadlock(create_mapper):
    queue = Queue()
    inputs = range(50)

    def as_iter(queue):
        while (item := queue.get()) is not None:
            yield item

    async def process(x):
        return x

    mapper = create_mapper(process, as_iter(queue), workers=10, loop=get_loop())
    it = mapper.iterate(timeout=4)
    for i in inputs:
        queue.put(i)

    # Check that we can get as many objects out as we put in, without deadlock
    result = []
    for _ in range(len(inputs)):
        result.append(next(it))
    if mapper.order_preserving:
        assert result == list(inputs)
    else:
        assert set(result) == set(inputs)

    # Check that iteration terminates cleanly
    queue.put(None)
    assert list(it) == []


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
@pytest.mark.parametrize("stop_at", [10, None])
def test_mapper_closes_iterable(create_mapper, stop_at):
    """Test that the iterable is closed when the `.iterate()` is closed or exhausted."""

    async def process(x):
        return x

    iterable_closed = False
    start_thread = None
    close_thread = None

    def gen():
        nonlocal iterable_closed, start_thread, close_thread
        start_thread = threading.get_ident()
        try:
            yield from range(50)
        finally:
            iterable_closed = True
            close_thread = threading.get_ident()

    mapper = create_mapper(process, gen(), workers=10, loop=get_loop())
    it = mapper.iterate()
    list(itertools.islice(it, stop_at))
    if stop_at is not None:
        it.close()
    assert iterable_closed
    assert start_thread == close_thread
    assert start_thread != threading.get_ident()


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
@settings(deadline=None)
@given(
    inputs=st.lists(st.integers(min_value=0, max_value=100), max_size=20),
    workers=st.integers(min_value=1, max_value=5),
)
def test_mapper_hypothesis(inputs, workers, create_mapper):
    async def process(input):
        await asyncio.sleep(input)
        return input

    loop = get_loop()
    mapper = create_mapper(process, inputs, workers=workers, loop=loop)
    with mock_time(loop):
        try:
            result = list(mapper.iterate(timeout=4))
        finally:
            join_all_tasks(loop)
    if mapper.order_preserving:
        assert result == inputs
    else:
        assert Counter(result) == Counter(inputs)


@pytest.mark.parametrize("create_mapper", [AsyncMapper, OrderedMapper])
@settings(deadline=None)
@given(
    inputs=st.lists(
        st.tuples(st.booleans(), st.integers(min_value=0, max_value=100)), max_size=20
    ),
    workers=st.integers(min_value=1, max_value=5),
)
def test_mapper_exception_hypothesis(inputs, workers, create_mapper):
    assume(any(n[0] for n in inputs))

    async def process(input):
        raising, n = input
        await asyncio.sleep(n)
        if raising:
            raise RuntimeError
        return input

    loop = get_loop()
    mapper = create_mapper(process, inputs, workers=workers, loop=loop)
    with mock_time(loop):
        try:
            with pytest.raises(RuntimeError):
                list(mapper.iterate(timeout=4))
        finally:
            join_all_tasks(loop)


def test_iter_over_async():
    async def gen():
        yield 1
        yield 2

    res = iter_over_async(gen(), get_loop())
    assert list(res) == [1, 2]


def test_iter_over_async_exception():
    async def gen():
        yield 1
        raise Exception("Some error")

    with pytest.raises(Exception) as excinfo:
        list(iter_over_async(gen(), get_loop()))
    assert str(excinfo.value) == "Some error"


def test_iter_over_async_empty():
    async def gen():
        for i in []:
            yield i

    res = iter_over_async(gen(), get_loop())
    assert list(res) == []
