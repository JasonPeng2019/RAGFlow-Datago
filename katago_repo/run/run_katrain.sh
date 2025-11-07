#!/usr/bin/env bash
set -euo pipefail

# Launch KaTrain from the project venv. Requires a GUI (local or X11 forwarding).
# Engine command to use inside KaTrain (Settings -> Engines) once the UI opens:
#   /scratch2/f004h1v/alphago_project/katago_repo/KataGo/cpp/build-opencl/katago gtp \
#     -model /scratch2/f004h1v/alphago_project/katago_repo/run/default_model.bin.gz \
#     -config /scratch2/f004h1v/alphago_project/katago_repo/run/default_gtp.cfg

exec /scratch2/f004h1v/alphago_project/venv/bin/katrain "$@"
