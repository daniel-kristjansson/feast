import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest
from pytest_mock import MockFixture

from feast import FeatureView, Field, FileSource
from feast.infra.offline_stores.snowflake import (
    SnowflakeOfflineStoreConfig,
    SnowflakeRetrievalJob,
    _get_feature_table_timestamp_bounds,
)
from feast.infra.offline_stores.snowflake_source import SnowflakeSource
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RepoConfig
from feast.types import Array, String


@pytest.fixture(params=["s3", "s3gov"])
def retrieval_job(request):
    offline_store_config = SnowflakeOfflineStoreConfig(
        type="snowflake.offline",
        account="snow",
        user="snow",
        password="snow",
        role="snow",
        warehouse="snow",
        database="FEAST",
        schema="OFFLINE",
        storage_integration_name="FEAST_S3",
        blob_export_location=f"{request.param}://feast-snowflake-offload/export",
    )
    retrieval_job = SnowflakeRetrievalJob(
        query="SELECT * FROM snowflake",
        snowflake_conn=MagicMock(),
        config=RepoConfig(
            registry="s3://ml-test/repo/registry.db",
            project="test",
            provider="snowflake.offline",
            online_store=SqliteOnlineStoreConfig(type="sqlite"),
            offline_store=offline_store_config,
            entity_key_serialization_version=3,
        ),
        full_feature_names=True,
        on_demand_feature_views=[],
    )
    return retrieval_job


def test_to_remote_storage(retrieval_job):
    stored_files = ["just a path", "maybe another"]
    with (
        patch.object(
            retrieval_job, "to_snowflake", return_value=None
        ) as mock_to_snowflake,
        patch.object(
            retrieval_job, "_get_file_names_from_copy_into", return_value=stored_files
        ) as mock_get_file_names_from_copy,
    ):
        assert retrieval_job.to_remote_storage() == stored_files, (
            "should return the list of files"
        )
        mock_to_snowflake.assert_called_once()
        mock_get_file_names_from_copy.assert_called_once_with(ANY, ANY)
        native_path = mock_get_file_names_from_copy.call_args[0][1]
        assert re.match("^s3://.*", native_path), "path should be s3://*"


def test_snowflake_to_df_internal(
    retrieval_job: SnowflakeRetrievalJob, mocker: MockFixture
):
    mock_execute = mocker.patch(
        "feast.infra.offline_stores.snowflake.execute_snowflake_statement"
    )
    mock_execute.return_value.fetch_pandas_all.return_value = pd.DataFrame.from_dict(
        {"feature1": ['["1", "2", "3"]', None, "[]"]}  # For Valid, Null, and Empty
    )

    feature_view = FeatureView(
        name="my-feature-view",
        entities=[],
        schema=[
            Field(name="feature1", dtype=Array(String)),
        ],
        source=FileSource(path="dummy.path"),  # Dummy value
    )
    retrieval_job._feature_views = [feature_view]
    retrieval_job._to_df_internal()


@pytest.mark.parametrize(
    "precision, max_value, expected",
    [
        (18, None, "NUMBER64"),
        (19, 42, "NUMBER32"),
        (19, 1234567890123, "NUMBER64"),
        (38, 1234567890123, "NUMBER64"),
    ],
)
def test_snowflake_source_integer_column_types(
    mocker: MockFixture, precision, max_value, expected
):
    config = RepoConfig(
        registry="s3://ml-test/repo/registry.db",
        project="test",
        provider="local",
        online_store=SqliteOnlineStoreConfig(type="sqlite"),
        offline_store=SnowflakeOfflineStoreConfig(
            type="snowflake.offline",
            account="snow",
            user="snow",
            password="snow",
            role="snow",
            warehouse="snow",
            database="FEAST",
            schema="PUBLIC",
        ),
        entity_key_serialization_version=3,
    )
    sample = MagicMock()
    sample.description = [
        SimpleNamespace(
            name="ID", type_code=0, precision=precision, scale=0, is_nullable=True
        )
    ]
    sample.fetch_pandas_all.return_value = pd.DataFrame({"ID": [1]})
    maximum = MagicMock()
    maximum.fetch_pandas_all.return_value = pd.DataFrame({"ID": [max_value]})
    mocker.patch("feast.infra.utils.snowflake.snowflake_utils.GetSnowflakeConnection")
    mocker.patch(
        "feast.infra.utils.snowflake.snowflake_utils.execute_snowflake_statement",
        side_effect=[sample, maximum],
    )
    source = SnowflakeSource(
        name="source", database="FEAST", schema="PUBLIC", table="T"
    )

    assert list(source.get_table_column_names_and_types(config)) == [("ID", expected)]


FALLBACK_RANGE = (
    datetime(2000, 1, 1, tzinfo=timezone.utc),
    datetime(2000, 1, 2, tzinfo=timezone.utc),
)


def _timestamp_bounds(row):
    conn = MagicMock()
    with patch(
        "feast.infra.offline_stores.snowflake.execute_snowflake_statement"
    ) as execute:
        execute.return_value.fetchone.return_value = row
        bounds = _get_feature_table_timestamp_bounds(
            conn, "entity_table", "event_timestamp", FALLBACK_RANGE
        )
    execute.assert_called_once_with(
        conn,
        'SELECT MIN("event_timestamp"), MAX("event_timestamp") FROM "entity_table"',
    )
    return bounds


def test_feature_table_timestamp_bounds_come_from_uploaded_entity_table():
    min_ts = datetime(2024, 3, 9, 16, 0, tzinfo=timezone.utc)
    max_ts = datetime(2024, 3, 10, 16, 0, 0, 123456, tzinfo=timezone.utc)

    assert _timestamp_bounds((min_ts, max_ts)) == (
        min_ts,
        max_ts + timedelta(microseconds=1),
    )


def test_feature_table_timestamp_bounds_fall_back_for_empty_entity_table():
    assert _timestamp_bounds((None, None)) == FALLBACK_RANGE
