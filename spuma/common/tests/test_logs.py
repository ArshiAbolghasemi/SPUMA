import json
import logging

from spuma.common.logs import JsonFormatter, set_run_id, setup_logging


def test_json_logging(tmp_path, config):
    log_file = setup_logging(config.logging.model_copy(update={"directory": tmp_path, "console": False}), "t")
    set_run_id("run-1")
    logging.getLogger("spuma.test").info("hello", extra={"agent_id": "solver"})
    for h in logging.getLogger().handlers:
        h.flush()
    record = json.loads(log_file.read_text().splitlines()[-1])
    assert record["message"] == "hello" and record["agent_id"] == "solver" and record["run_id"] == "run-1"
    assert isinstance(JsonFormatter().format(logging.makeLogRecord({"msg": "x"})), str)
