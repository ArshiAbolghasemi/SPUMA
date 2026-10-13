from langchain_core.messages import HumanMessage

from spuma.models.benchmark import benchmark_backend


def test_benchmark_reports_all_measurements(tiny_backend, config, set_settings):
    generation = config.model.generation.model_copy(update={"max_new_tokens": 4})
    set_settings(config.replace(model=config.model.model_copy(update={"generation": generation})))
    prompts = [tiny_backend.render([HumanMessage(q)], None) for q in ("Solve for x", "Review the solution")]
    r = benchmark_backend(tiny_backend, prompts)
    assert r["backend"] == "transformers" and r["requests"] == 2
    assert r["generation"]["tokens_per_s"] > 0 and r["generation"]["generated_tokens"] > 0
    assert r["teacher_forced_scoring"]["seconds"] > 0 and r["semantic_sampling"]["n"] == 10
