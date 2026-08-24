from __future__ import annotations

import asyncio
import logging
import signal

from mormi_api.db import Database
from mormi_api.ladder_analysis_repository import LadderAnalysisRepository
from mormi_api.ladder_analysis_worker import DatabaseSpeechLoader, LadderAnalysisWorker
from mormi_api.ladder_model.runtime import LadderModelRuntime
from mormi_api.security import StoredTextCodec
from mormi_api.settings import get_settings


async def run_worker() -> None:
    settings = get_settings()
    if not settings.ladder_analysis_enabled or settings.ladder_model_dir is None:
        raise RuntimeError("LADDER_WORKER_CONFIGURATION_INVALID")

    database = Database(settings.database_url)
    worker = LadderAnalysisWorker(
        LadderAnalysisRepository(
            database,
            lease_seconds=settings.ladder_analysis_lease_seconds,
        ),
        LadderModelRuntime(settings.ladder_model_dir),
        load_speech=DatabaseSpeechLoader(
            database,
            StoredTextCodec(settings.raw_data_encryption_key),
        ),
        poll_interval_seconds=settings.ladder_analysis_poll_interval_seconds,
        batch_size=settings.ladder_analysis_batch_size,
    )
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_requested.set)

    worker_task = asyncio.create_task(worker.run_forever(), name="ladder-analysis-worker")
    stop_task = asyncio.create_task(stop_requested.wait(), name="ladder-worker-stop")
    try:
        done, _ = await asyncio.wait(
            {worker_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker_task in done:
            await worker_task
    finally:
        stop_task.cancel()
        await worker.stop()
        await worker_task
        await database.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
