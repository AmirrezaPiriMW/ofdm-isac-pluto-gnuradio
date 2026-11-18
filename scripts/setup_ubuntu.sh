#!/usr/bin/env bash
# One-time set-up on Ubuntu 22.04 / 24.04 (Debian-based systems work the same way).
# Installs GNU Radio 3.10 (incl. gr-iio for the PlutoSDR), libiio with its Python bindings and the
# Python packages used by this repository from the distribution, then creates a virtual environment
# that also sees GNU Radio's and libiio's Python bindings, and installs pyadi-iio into it.
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y gnuradio libiio-utils python3-libiio python3-venv python3-pip \
    python3-numpy python3-scipy python3-yaml python3-matplotlib python3-pytest
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[test]"
# pyadi-iio without its pylibiio dependency: the newest pylibiio wheels need libiio 1.x, while
# Ubuntu ships libiio 0.2x; the matching bindings come from the python3-libiio package above.
.venv/bin/pip install --no-deps pyadi-iio
.venv/bin/python -c "import iio, adi; print('libiio', iio.version, '- pyadi-iio OK')"
echo
echo "Done. Activate with:   source .venv/bin/activate"
echo "Check the Pluto with:  iio_info -s"
echo "Then run:              make test && make demo"
