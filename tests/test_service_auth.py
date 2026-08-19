from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, Request

from mormi_api.main import require_service_key
from mormi_api.settings import Settings


def request_with_key(expected_key: str) -> Request:
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=Settings(service_api_key=expected_key),
        )
    )
    return cast(Request, SimpleNamespace(app=app))


def test_service_key_accepts_the_shared_backend_key() -> None:
    require_service_key(request_with_key("shared-secret"), "shared-secret")


def test_service_key_rejects_a_different_backend_key() -> None:
    with pytest.raises(HTTPException) as raised:
        require_service_key(request_with_key("shared-secret"), "different-secret")

    assert raised.value.status_code == 401


def test_production_runtime_does_not_require_a_legacy_encryption_key() -> None:
    settings = Settings(
        environment="production",
        database_url="postgresql+asyncpg://user:password@database:5432/mormi",
        service_api_key="shared-secret",
        raw_data_encryption_key=None,
    )

    settings.validate_runtime_safety()
