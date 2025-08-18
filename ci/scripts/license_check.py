#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import textwrap
from datetime import datetime
from pathlib import Path


year = datetime.now().year

license_text = textwrap.dedent("""\
    SPDX-License-Identifier: LicenseRef-Apache2

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """)

default_copyright_text = (
    f"SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)


def process_file(filepath: Path, dry_run: bool):
    comment_start = get_comment_delimiter(filepath)
    lines = filepath.read_text().splitlines()

    start_line = 0
    if lines[0].startswith("#!"):
        # Make sure there's a blank line after the shebang
        if lines[1] != "":
            lines.insert(1, "")
        start_line = 2

    license_block = []
    for line in lines[start_line:]:
        if line.startswith(comment_start):
            license_block.append(line)
        else:
            break


def get_comment_delimiter(filepath: Path) -> str:
    match filepath.suffix:
        case ".py":
            return "#"
        case ".rs":
            return "//"
        case _:
            raise ValueError(f"Unsupported file type: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Ensure files have proper license headers")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")

    args = parser.parse_args()

    modified_count = 0
    for filename in args.files:
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        process_file(filepath, args.dry_run)


if __name__ == "__main__":
    main()
