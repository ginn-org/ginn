#!/usr/bin/env python3

import argparse
import re
import sys
from typing import List

HEADER = """ Copyright.*

 Licensed under the Apache License, Version 2\.0 \(the "License"\);
 you may not use this file except in compliance with the License\.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied\.
 See the License for the specific language governing permissions and
 limitations under the License\."""


CPP_HEADER_MATCHER = re.compile(
    "(" + "\n".join(["//" + line for line in HEADER.split("\n")]) + "\n\n)(.*)$",
    re.DOTALL,
)
PY_HEADER_MATCHER = re.compile(
    "(" + "\n".join(["#" + line for line in HEADER.split("\n")]) + "\n\n)(.*)$",
    re.DOTALL,
)


NOTICE = """ Copyright 2022 Bloomberg Finance L.P.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""

CPP_NOTICE = "\n".join(["//" + line for line in NOTICE.split("\n")]) + "\n\n"
PY_NOTICE = "\n".join(["#" + line for line in NOTICE.split("\n")]) + "\n\n"


def infer_lang(fname: str) -> str:
    if fname.endswith(".py"):
        return "py"
    for ext in [".h", ".cu", ".cpp", ".c"]:
        return "cpp"
    raise f"Failed to infer language from filename {fname}!"


def add_notice_to_files(filenames: List[str], lang: str, write: bool = False):
    for file in filenames:
        lang_ = infer_lang(file) if lang == "auto" else lang
        if lang_ == "cpp":
            matcher = CPP_HEADER_MATCHER
            notice = CPP_NOTICE
        elif lang_ == "py":
            matcher = PY_HEADER_MATCHER
            notice = PY_NOTICE
        else:
            raise f"Unexpected language: {lang}!"

        with open(file) as f:
            data = f.read()
        match = re.match(matcher, data)
        if match:
            subbed = re.sub(matcher, notice + r"\2", data)
        else:
            subbed = notice + data
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
    parser.add_argument(
        "-l",
        "--lang",
        default="auto",
        help="language of the files (auto infers from extension)",
        choices=["auto", "cpp", "py"],
    )
    args = parser.parse_intermixed_args()

    add_notice_to_files(args.files, args.lang, args.write)
