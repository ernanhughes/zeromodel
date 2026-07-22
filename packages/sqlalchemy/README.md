# zeromodel-sqlalchemy 1.0.13

`zeromodel-sqlalchemy` owns the SQLAlchemy and SQLite persistence adapter for
the validated video action-set domain. Its import namespace is
`zeromodel.persistence.sqlalchemy`.

The package depends on `zeromodel==1.0.13`, `zeromodel-video==1.0.13`, NumPy,
and SQLAlchemy 2.x. It does not depend on analysis, vision, research fixtures,
examples, Torch, TorchVision, Transformers, or Pillow.

## Boundary

Video owns DTOs, domain services, runtime protocols, and the in-memory Store.
This package owns SQLAlchemy engines, sessions, ORM rows, schema creation, and
SQL Store implementations. Store methods accept and return video DTOs; ORM
objects remain internal to `zeromodel.persistence.sqlalchemy.db.orm` and
`zeromodel.persistence.sqlalchemy.db.stores`.

## Schema Initialization

Database creation is explicit. Importing `zeromodel.persistence.sqlalchemy`
does not create files, tables, engines, or sessions.

```python
from pathlib import Path

from zeromodel.persistence.sqlalchemy import (
    SqlAlchemyVideoActionSetStore,
    create_database_engine,
    create_schema,
    create_session_factory,
    sqlite_database_url,
)

engine = create_database_engine(sqlite_database_url(Path("video.sqlite")))
create_schema(engine)
store = SqlAlchemyVideoActionSetStore(create_session_factory(engine))
```

For runtime composition:

```python
from zeromodel.persistence.sqlalchemy import build_sqlite_runtime

runtime = build_sqlite_runtime("sqlite:///video.sqlite", initialize_schema=True)
```

## Validation

Package-local tests cover import isolation, explicit schema creation, SQLite
foreign keys, in-memory Store parity for identity and episode plans,
transaction rollback on batch conflicts, MatrixBlob deduplication/conflict
handling, observation operation filters, database reopen behavior, tamper
detection, and runtime composition.
