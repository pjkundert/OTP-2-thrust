#
# Makefile for OTP-2-thrust project
# Based on python-slip39 GNUmakefile structure
#

SHELL		= /bin/bash

# Python detection - prefer python3
PYTHON		?= $(shell python3 --version >/dev/null 2>&1 && echo python3 || echo python )
PYTHON_P	= $(shell which $(PYTHON))
PYTHON_V	= $(shell $(PYTHON) -c "import sys; print('-'.join((('venv' if sys.prefix != sys.base_prefix else next(iter(filter(None,sys.base_prefix.split('/'))))),sys.platform,sys.implementation.cache_tag)))" 2>/dev/null )

# VirtualEnv configuration
VENV_DIR	= $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/.. )
VENV_NAME	= OTP-2-thrust-$(PYTHON_V)
VENV		= $(VENV_DIR)/$(VENV_NAME)
VENV_OPTS	= # --copies

# Nix options
NIX_OPTS	?= # --pure

.PHONY: all help clean venv FORCE

all:			help

help:
	@echo "Makefile for OTP-2-thrust. Targets:"
	@echo "  help			This help"
	@echo "  venv			Create and activate Python virtual environment"
	@echo "  venv-%			Run make target % in virtual environment"
	@echo "  nix-%			Run make target % in nix environment"
	@echo "  install		Install Python dependencies"
	@echo "  plot			Run the satellite data plotting script"
	@echo "  clean			Remove build artifacts and virtual environments"

# VirtualEnv targets - based on python-slip39 pattern
venv-%:			$(VENV)
	@echo; echo "*** Running in $< VirtualEnv: make $*"
	@bash --init-file $</bin/activate -ic "make $*"

venv:			$(VENV)
	@echo; echo "*** Activating $< VirtualEnv for Interactive $(SHELL)"
	@bash --init-file $</bin/activate -i

$(VENV):
	@[[ "$(PYTHON_V)" =~ "^venv" ]] && ( echo -e "\n\n!!! $@ Cannot start a venv within a venv"; false ) || true
	@echo; echo "*** Building $@ VirtualEnv..."
	@rm -rf $@ && $(PYTHON) -m venv $(VENV_OPTS) $@ && sed -i -e '1s:^:. $$HOME/.bashrc\n:' $@/bin/activate \
	    && source $@/bin/activate \
	    && $(PYTHON) -m pip install --upgrade pip \
	    && make install

# Nix targets - based on python-slip39 pattern
nix-%:
	@if [ -r flake.nix ]; then \
	    nix develop $(NIX_OPTS) --command make $*; \
        else \
	    nix-shell $(NIX_OPTS) --run "make $*"; \
	fi

# Install Python dependencies
install:
	$(PYTHON) -m pip install pandas matplotlib numpy

# Run the plotting script
plot:
	$(PYTHON) plot_satellite_data.py

clean:
	@rm -rf $(VENV) *.png __pycache__ *.pyc

# Print make variables
print-%:
	@echo $* = $($*)
	@echo $*\'s origin is $(origin $*)