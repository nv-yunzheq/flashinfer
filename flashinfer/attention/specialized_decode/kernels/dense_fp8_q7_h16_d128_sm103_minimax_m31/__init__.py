# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Specialized decode kernel ``dense_fp8_q7_h16_d128_sm103_minimax_m31``: 7 dense FP8
query tokens, 16 query heads over 1 K/V head, head_dim 128, page 128, on SM103 (the
MiniMax-M3.1 draft-model attention).

``kernel.py`` holds the case and ``auto`` guards, the staging, the lazy compile, the launch,
the stock path and the diagnostics; ``workloads.json`` is the measured envelope ``auto``
admits. The CuTe DSL program lives in
``flashinfer/cute_dsl/attention/gqa_decode_cluster_q7.py`` and is imported by the first plan.
"""

from .kernel import DenseFp8Q7Sm103MiniMaxM31Kernel as KERNEL
from .kernel import stats

__all__ = ["KERNEL", "stats"]
