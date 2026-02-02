# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from functools import total_ordering

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError


def get_pkg_version(pkg_name):
    try:
        version_str = version(pkg_name)
    except PackageNotFoundError:
        try:
            from importlib import import_module
            version_str = import_module(pkg_name).__version__
        except Exception:
            version_str = None

    return version_str


@total_ordering
class Version:
    VERSION_REGEX = re.compile(
        r"(?P<major>\d{1,4})\.(?P<minor>\d{1,4})"
        r"(?:\.(?P<patch>\d{1,4})|\.rc(?P<rc>\d{1,4})(?:\.b(?P<beta>\d{1,4}))?|\.rc(?P<rc_only>\d{1,4}))?"
    )

    def __init__(self, version_str):
        m = self._parse_version_str(version_str)
        self._major = m.group("major")
        self._minor = m.group("minor")
        self._patch = m.group("patch")
        self._rc = m.group("rc") or m.group("rc_only")
        self._beta = m.group("beta")

    def __repr__(self):
        base = f"{self._major}.{self._minor}"
        if self._patch is not None:
            return f"{base}.{self._patch}"
        if self._rc is not None:
            return f"{base}.rc{self._rc}" + (f".b{self._beta}" if self._beta else "")
        return base

    def __eq__(self, other):
        return self.cmp_tuple() == self._other_version(other)

    def __lt__(self, other):
        return self.cmp_tuple() < self._other_version(other)

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def patch(self):
        return self._patch

    @property
    def rc(self):
        return self._rc

    @property
    def beta(self):
        return self._beta

    @staticmethod
    def _parse_version_str(version_str):
        m = Version.VERSION_REGEX.match(version_str)
        if m:
            return m
        ver = get_pkg_version(version_str)
        if ver:
            return Version._parse_version_str(ver)

        raise ValueError("Invalid version string: {}".format(version_str))

    @staticmethod
    def _other_version(other):
        if isinstance(other, Version):
            return other.cmp_tuple()

        return Version(other).cmp_tuple()

    def cmp_tuple(self):
        return (
            int(self._major),
            int(self._minor),
            int(self._patch) if self._patch else 0,
            int(self._rc) if self._rc else float("inf"),
            int(self._beta) if self._beta else float("inf")
        )
