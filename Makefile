.PHONY: install install-isaacsim test benchmark benchmark-action benchmark-garment \
        benchmark-all download-data download-checkpoints lint format clean help

PYTHON  ?= python
SIM     ?= pybullet
MODE    ?= fixed_point
SAMPLE  ?= green_tshirt/grasp/02
GARMENT ?= green_tshirt
ACTION  ?= grasp
ROBOT   ?= piper

help:
	@echo "RGBench targets:"
	@echo
	@echo "  Install & data"
	@echo "    make install                 Editable install with [all] extras into the active venv"
	@echo "    make install-isaacsim        Install extras into Isaac Sim's bundled python.sh"
	@echo "    make download-checkpoints    Pull SAM + GroundingDINO checkpoints"
	@echo "    make download-data           Pull the full Hugging Face dataset (6.7 GB)"
	@echo "    make download-data SAMPLE_ONLY=1   Pull just one smoke-test capture (~100 MB)"
	@echo
	@echo "  Run benchmark (four scopes)"
	@echo "    make benchmark           SIM=pybullet SAMPLE=green_tshirt/grasp/02"
	@echo "                                 One specific (garment, action, sample) cell"
	@echo "    make benchmark-action    SIM=pybullet GARMENT=green_tshirt ACTION=grasp"
	@echo "                                 All samples of one (garment, action)"
	@echo "    make benchmark-garment   SIM=pybullet GARMENT=green_tshirt"
	@echo "                                 All actions and samples of one garment"
	@echo "    make benchmark-all       SIM=pybullet"
	@echo "                                 Every cell in experiment_library.yaml (~98)"
	@echo "    Common options:   MODE={fixed_point,robot}  ROBOT=piper"
	@echo "                      SIM={pybullet,isaacsim,mujoco,garment_dynamics}"
	@echo
	@echo "  Compare & analyse"
	@echo "    python scripts/compare_to_paper.py outputs/ --metric cd_l1_r2s"
	@echo "                                 Rank your runs against paper baselines"
	@echo
	@echo "  Dev"
	@echo "    make test                    Run tests/"
	@echo "    make lint                    Static checks"
	@echo "    make clean                   Remove build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip wheel
	$(PYTHON) -m pip install -e ".[all]"

install-isaacsim:
	@if [ -z "$$ISAACSIM_PYTHON" ]; then \
	  echo "error: set ISAACSIM_PYTHON to your isaacsim python.sh"; exit 1; fi
	"$$ISAACSIM_PYTHON" -m pip install -r requirements-isaacsim.txt

download-checkpoints:
	$(PYTHON) scripts/download_checkpoints.py

download-data:
ifdef SAMPLE_ONLY
	$(PYTHON) scripts/download_data.py --sample-only
else
	$(PYTHON) scripts/download_data.py
endif

# Single cell:  SAMPLE=cloth/action/idx
benchmark:
	@cloth=$$(echo $(SAMPLE) | cut -d/ -f1); \
	action=$$(echo $(SAMPLE) | cut -d/ -f2); \
	idx=$$(echo $(SAMPLE) | cut -d/ -f3); \
	$(PYTHON) scripts/run_benchmark.py \
	  params.sim_environment=$(SIM) \
	  params.sim_mode=$(MODE) \
	  params.cloth_name=$$cloth \
	  params.action_type=$$action \
	  params.sample_index=$$idx \
	  params.robot=$(ROBOT) \
	  env=$(SIM) \
	  cloth_params=$$cloth

# All samples of one (garment, action)
benchmark-action:
	$(PYTHON) scripts/run_batch.py \
	  --sim $(SIM) --mode $(MODE) \
	  --garment $(GARMENT) --action $(ACTION) --robot $(ROBOT)

# All actions and samples of one garment
benchmark-garment:
	$(PYTHON) scripts/run_batch.py \
	  --sim $(SIM) --mode $(MODE) \
	  --garment $(GARMENT) --robot $(ROBOT)

# Every cell in experiment_library.yaml
benchmark-all:
	$(PYTHON) scripts/run_batch.py \
	  --sim $(SIM) --mode $(MODE) --robot $(ROBOT)

test:
	$(PYTHON) -m pytest tests/ -q || $(PYTHON) -m unittest discover tests

lint:
	$(PYTHON) -m ruff check rgbench scripts tools tests || true

format:
	$(PYTHON) -m ruff format rgbench scripts tools tests || true

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
