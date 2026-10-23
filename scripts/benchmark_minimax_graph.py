"""Benchmark one complete MiniMax verification CUDA graph across four ranks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualify.minimax_graph import main

if __name__ == "__main__":
    main()
