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

"""Kernel registry of the specialized decode backend.

Kernels are found, not listed: every subpackage of ``kernels/`` whose name does not start
with ``_`` must export ``KERNEL``, a :class:`SpecializedDecodeKernel` subclass whose
``name`` is the subpackage name and which declares its compute capabilities. Adding a
kernel therefore adds one subpackage and edits no shared file. Kernel modules import no
device code at module scope, so discovery is cheap. A subpackage that fails to import or
to declare its kernel is skipped with a warning and recorded in
:func:`discovery_errors`; the other kernels keep working.

Order: kernels are asked in name order, and the first one that admits a call serves it.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from types import MappingProxyType
from typing import Dict, Mapping, Type

from ...jit.core import logger
from . import kernels as _kernels_package
from .kernel import SpecializedDecodeKernel

_discovery_errors: Dict[str, str] = {}


@functools.cache
def registered_kernels() -> Mapping[str, Type[SpecializedDecodeKernel]]:
    """The registered kernel classes by name, in name order (read-only)."""
    found: Dict[str, Type[SpecializedDecodeKernel]] = {}
    modules = sorted(
        pkgutil.iter_modules(_kernels_package.__path__), key=lambda info: info.name
    )
    for info in modules:
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{_kernels_package.__name__}.{info.name}")
            kernel = getattr(module, "KERNEL", None)
            if not (
                isinstance(kernel, type) and issubclass(kernel, SpecializedDecodeKernel)
            ):
                raise TypeError("KERNEL is not a SpecializedDecodeKernel subclass")
            if kernel.name != info.name:
                raise ValueError(
                    f"KERNEL.name {kernel.name!r} differs from its subpackage name"
                )
            if not kernel.compute_capabilities:
                raise ValueError("KERNEL must declare compute_capabilities")
        except Exception as exc:  # one broken kernel must not disable the others
            _discovery_errors[info.name] = f"{type(exc).__name__}: {exc}"[:2000]
            logger.warning(
                "BatchDecodeWithPagedKVCacheWrapper specialized backend: skipping "
                f"kernel {info.name!r} ({_discovery_errors[info.name]})"
            )
            continue
        found[info.name] = kernel
    return MappingProxyType(found)


def discovery_errors() -> Dict[str, str]:
    """Subpackages of ``kernels/`` that could not be registered, with the reason."""
    registered_kernels()
    return dict(_discovery_errors)
