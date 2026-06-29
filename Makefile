.PHONY: clean clean-dry-run run pdf-pages extract extract-force auto auto-force

UV_RUN := uv run

DPI ?= 200
MODEL ?= qwen2.5vl:7b
TEXT_MODEL ?= qwen2.5:7b
VISION_MODEL ?= qwen2.5vl:7b
NUM_CTX ?= 16384
NUM_PREDICT ?= 2500
TIMEOUT_SECONDS ?= 1000

PDF_PAGES_ARGS := --force --dpi $(DPI)
EXTRACT_ARGS := --model $(MODEL) --num-ctx $(NUM_CTX) --num-predict $(NUM_PREDICT) --timeout-seconds $(TIMEOUT_SECONDS)
AUTO_ARGS := --text-model $(TEXT_MODEL) --vision-model $(VISION_MODEL) \
             --num-ctx $(NUM_CTX) --num-predict $(NUM_PREDICT) \
             --timeout-seconds $(TIMEOUT_SECONDS) --dpi $(DPI)

ifdef ONLY
PDF_PAGES_ARGS += --only "$(ONLY)"
EXTRACT_ARGS += --only "$(ONLY)"
AUTO_ARGS += --only "$(ONLY)"
endif

ifdef MAX_PAGES
EXTRACT_ARGS += --max-pages $(MAX_PAGES)
AUTO_ARGS += --max-pages $(MAX_PAGES)
endif

clean:
	$(UV_RUN) python -m scripts.clean_outputs && rm -rf all-report.txt

clean-dry-run:
	$(UV_RUN) python -m scripts.clean_outputs --dry-run

pdf-pages:
	$(UV_RUN) python -m scripts.pdf_to_pages $(PDF_PAGES_ARGS)

# Ancien pipeline vision-only (conservé)
extract:
	$(UV_RUN) python -m scripts.extract_vision_ollama $(EXTRACT_ARGS)

extract-force:
	$(UV_RUN) python -m scripts.extract_vision_ollama $(EXTRACT_ARGS) --no-skip-existing

# Nouveau pipeline orchestrateur (texte ou vision selon le PDF)
auto:
	$(UV_RUN) python -m scripts.extract $(AUTO_ARGS)

auto-force:
	$(UV_RUN) python -m scripts.extract $(AUTO_ARGS) --no-skip-existing --force-render

compile:
	touch all-report.txt && for f in reports/*.txt; do cat "$$f" >> all-report.txt; done

# run utilise désormais l'orchestrateur (plus besoin de pdf-pages séparé)
run: auto
