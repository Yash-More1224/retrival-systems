.PHONY: all data retrieval eval ablation submit test clean

all: data test retrieval eval ablation submit

data:
	python build_pipeline.py --datasets mind ebnerd --config config/default.yaml

retrieval:
	python -m src.retrieval.run_bm25_eval --datasets mind ebnerd
	python -m src.retrieval.build_embeddings --datasets mind ebnerd
	python -m src.retrieval.run_semantic_eval --datasets mind ebnerd

eval:
	python -m src.eval.run_eval --datasets mind ebnerd --split test

ablation:
	python -m src.eval.run_ablation --split test

submit:
	python -m src.submission.mind
	python -m src.submission.ebnerd

test:
	pytest -q tests/

clean:
	rm -rf data/interim data/splits feature_store results
