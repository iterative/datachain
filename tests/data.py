from datetime import datetime, timezone

from datachain.node import Entry

utc = timezone.utc
TIME_ZERO = datetime.fromtimestamp(0, tz=utc)

ENTRIES = [
    Entry.from_file(
        parent="",
        name="description",
        etag="60a7605e934638ab9113e0f9cf852239",
        version="7e589b7d-382c-49a5-931f-2b999c930c5e",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=13,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="cats",
        name="cat1",
        etag="4a4be40c96ac6314e91d93f38043a634",
        version="309eb4a4-bba9-47c1-afcd-d7c51110af6f",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="cats",
        name="cat2",
        etag="0268c692ff940a830e1e7296aa48c176",
        version="f9d168d3-6d1b-47ef-8f6a-81fce48de141",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs",
        name="dog1",
        etag="8fdb60801e9d39a5286aa01dd1f4f4f3",
        version="b9c31cf7-d011-466a-bf16-cf9da0cb422a",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs",
        name="dog2",
        etag="2d50c921b22aa164a56c68d71eeb4100",
        version="3a8bb6d9-38db-47a8-8bcb-8972ea95aa20",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=3,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs",
        name="dog3",
        etag="33c6c2397a1b079e903c474df792d0e2",
        version="ee49e963-36a8-492a-b03a-e801b93afb40",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs/others",
        name="dog4",
        etag="a5e1a5d93ff242b745f5cf87aeb726d5",
        version="c5969421-6900-4060-bc39-d54f4a49b9fc",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
]

# files with directory name collisions:
#   dogs/others/
#   dogs/others
#   dogs/
INVALID_ENTRIES = [
    Entry.from_file(
        parent="dogs/others",
        name="",
        etag="68b329da9893e34099c7d8ad5cb9c940",
        version="85969421-6900-4060-bc39-d54f4a49b9ab",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs",
        name="others",
        etag="68b329da9893e34099c7d8ad5cb9c940",
        version="85969421-6900-4060-bc39-d54f4a49b9ab",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
    Entry.from_file(
        parent="dogs",
        name="",
        etag="68b329da9893e34099c7d8ad5cb9c940",
        version="85969421-6900-4060-bc39-d54f4a49b9ab",
        is_latest=True,
        last_modified=datetime(2023, 2, 27, 18, 28, 54, tzinfo=utc),
        size=4,
        owner_name="webfile",
        owner_id="75aa57f09aa0c8caeab4f8c24e99d10f8e7faeebf76c078efc7c6caea54ba06a",
    ),
]
