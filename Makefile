# -------------------------
# VARIABLES
# -------------------------
PYTHON=python

# -------------------------
# RUN API
# -------------------------
run:
	$(PYTHON) run_api.py

# -------------------------
# TRAIN MODEL
# -------------------------
train:
	$(PYTHON) scripts/train.py

# -------------------------
# EVALUATE MODEL
# -------------------------
evaluate:
	$(PYTHON) scripts/evaluate.py

# -------------------------
# EXPLAIN (SHAP)
# -------------------------
explain:
	$(PYTHON) scripts/explain.py

# -------------------------
# RUN ALL
# -------------------------
all:
	$(PYTHON) scripts/run_all.py

# -------------------------
# TESTS
# -------------------------
test:
	pytest tests/

test-api:
	pytest app/tests/

# -------------------------
# FORMAT (optional)
# -------------------------
format:
	black .

# -------------------------
# CLEAN
# -------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete