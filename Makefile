.PHONY: train audit predict test lint demo verify submission
train:
	python -m water_quality train --config configs/default.json
audit:
	python -m water_quality audit
predict:
	python -m water_quality predict --input examples/measurements.csv
test:
	python -m pytest -q
lint:
	ruff check src tests scripts app.py
demo:
	streamlit run app.py
verify:
	python scripts/verify_submission.py
submission:
	python scripts/build_submission.py
