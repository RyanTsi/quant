"""CLI wrapper for qlib dump utilities.

This module intentionally keeps CLI behavior stable while delegating reusable
implementation to runtime adapter modules.
"""

from __future__ import annotations

import fire

from runtime.adapters.dump_bin_core import DumpDataAll, DumpDataFix, DumpDataUpdate


def main() -> None:
    fire.Fire({"dump_all": DumpDataAll, "dump_fix": DumpDataFix, "dump_update": DumpDataUpdate})


if __name__ == "__main__":
    main()
