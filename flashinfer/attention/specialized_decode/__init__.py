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

"""``backend="specialized"`` of :class:`flashinfer.BatchDecodeWithPagedKVCacheWrapper`.

One decode backend for kernels generated for one served decode case each (a model's
attention geometry, dtypes, mask, page size, batch and KV-length ranges on one GPU
architecture). ``kernel.py`` defines the kernel contract, the plan description the
kernels judge and the kill switch; ``registry.py`` finds the kernels (one self-contained
subpackage of ``kernels/`` each); ``backend.py`` holds the wrapper's plan / run state,
the ``auto`` decision and :func:`specialized_decode_stats`.

Dispatch happens at ``plan()``: the kernels are asked in name order and the first one
that admits the call serves it (``auto``: its case and its measured envelope; an explicit
``backend="specialized"``: its case, else ``NotImplementedError``). ``run()`` launches
what the plan chose; under ``use_cuda_graph=True`` the choice is part of the frozen plan.
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``, read once per plan by the backend, keeps
``auto`` off every kernel and runs an explicit plan on the kernel's stock path.
"""

from .backend import (
    SpecializedDecode,
    auto_kernel,
    plan_kernel,
    specialized_decode_stats,
)
from .kernel import KILL_SWITCH_ENV, DecodePlanCall, SpecializedDecodeKernel
from .registry import discovery_errors, registered_kernels

__all__ = [
    "KILL_SWITCH_ENV",
    "DecodePlanCall",
    "SpecializedDecode",
    "SpecializedDecodeKernel",
    "auto_kernel",
    "discovery_errors",
    "plan_kernel",
    "registered_kernels",
    "specialized_decode_stats",
]
