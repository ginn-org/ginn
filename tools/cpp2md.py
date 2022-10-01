#!/usr/bin/env python3

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BlockKind(Enum):
    TEXT = 1
    CODE = 2


@dataclass
class Block:
    kind: BlockKind
    content: str

    @staticmethod
    def _trim_lines(s: str) -> str:
        maybe_last_newline = "\n" if s.endswith("\n") else ""
        lines = s.split("\n")
        start = 0
        end = 0
        while start < len(lines) and len(lines[start]) == 0:
            start += 1
        while end > -len(lines) and len(lines[end - 1]) == 0:
            end -= 1
        return "\n".join(lines[start:end]) + maybe_last_newline

    def __str__(self):
        if self.kind == BlockKind.TEXT:
            return self.content
        elif self.kind == BlockKind.CODE:
            assert self.content.endswith("\n")
            return f"```cpp\n{self._trim_lines(self.content)}```\n"
        else:
            raise ValueError("Unexpected block kind {self.kind}")

    def empty(self) -> bool:
        return self.content == "" or self.content == "\n"


if __name__ == "__main__":
    prev_block: Optional[Block] = None

    if len(sys.argv) > 1:
        ins = open(sys.argv[1], "r")
    else:
        ins = sys.stdin

    for line in ins:
        line_ = line.lstrip()
        if line_.startswith("///"):
            # md text block
            kind = BlockKind.TEXT
            content = line_[3:].lstrip(" \t")
        else:
            # md cpp code block
            kind = BlockKind.CODE
            content = line
        if prev_block and prev_block.kind == kind:
            prev_block.content += content
        else:
            # Starting a new block, print the old one
            if prev_block and not prev_block.empty():
                sys.stdout.write(str(prev_block) + "\n")
            prev_block = Block(kind, content)

    if not prev_block.empty():
        sys.stdout.write(str(prev_block))

    # Put footnote at the end
    my_name = sys.argv[0]
    fname = sys.argv[1] if len(sys.argv) > 1 else "stdin"
    sys.stdout.write(f"\n---\n\n(Generated with `{my_name}` from `{fname}`.)\n")
