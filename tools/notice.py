#!/usr/bin/env python3

import argparse
import re
import sys
from typing import List

CPP_HEADER_RE = re.compile(
    "(// Copyright.*\n"
    "//\n"
    '// Licensed under the Apache License, Version 2\.0 \(the "License"\);\n'
    "// you may not use this file except in compliance with the License\.\n"
    "// You may obtain a copy of the License at\n"
    "//\n"
    "//     http://www.apache.org/licenses/LICENSE-2.0\n"
    "//\n"
    "// Unless required by applicable law or agreed to in writing, software\n"
    '// distributed under the License is distributed on an "AS IS" BASIS,\n'
    "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied\.\n"
    "// See the License for the specific language governing permissions and\n"
    "// limitations under the License\.\n\n)"
    "(.*)$",  # rest of the file
    re.DOTALL,
)

CPP_NOTICE = """// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"""


def add_notice_to_files(filenames: List[str], write: bool = False):
    for file in filenames:
        with open(file) as f:
            data = f.read()
        match = re.match(CPP_HEADER_RE, data)
        if match:
            subbed = re.sub(CPP_HEADER_RE, CPP_NOTICE + r"\2", data)
        else:
            subbed = CPP_NOTICE + data
        if write:
            with open(file, "w") as f:
                f.write(subbed)
        else:
            sys.stdout.write(subbed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="files to add the notice to")
    parser.add_argument(
        "-w",
        "--write",
        required=False,
        action="store_true",
        help="overwrite files with new notice",
    )
    args = parser.parse_intermixed_args()

    add_notice_to_files(args.files, args.write)
