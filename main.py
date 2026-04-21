"""
SPX v7.5.0 [Central Entry Point]
Unified CLI for compression, decompression, and benchmarking.
Usage:
    python main.py compress [path] [--optimize]
    python main.py decompress [path] [--output path]
    python main.py benchmark [data_path] [--num_tests N]
"""

import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ZSPX Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Compress
    parser_comp = subparsers.add_parser("compress", help="Compress an image")
    parser_comp.add_argument("path", help="Path to input image")
    parser_comp.add_argument("--optimize", action="store_true", help="Enable heuristic optimization")
    parser_comp.add_argument("--bitplane", action="store_true", help="Enable experimental Bitplane engine")
    parser_comp.add_argument("--raw-res", help="Resolution for RAW files (e.g. 1920x1080)")
    parser_comp.add_argument("--raw-mode", choices=["RGB", "L"], default="RGB", help="Color mode for RAW files")

    # Decompress
    parser_decomp = subparsers.add_parser("decompress", help="Decompress an SPX bitstream")
    parser_decomp.add_argument("path", help="Path to .spx file")
    parser_decomp.add_argument("--output", help="Optional output path")

    # Benchmark (Comparative)
    parser_bench = subparsers.add_parser("benchmark", help="Run comparative benchmark hub")
    parser_bench.add_argument("path", nargs="?", default="./data/gold", help="Dataset path or alias")
    parser_bench.add_argument("--num_tests", "-n", type=int, help="Limit number of tests")
    parser_bench.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser_bench.add_argument("--codec", choices=["spx", "webp", "jxl", "bench"], default="spx", help="Codec to test")
    parser_bench.add_argument("--offset", type=int, help="Skip first N images")
    parser_bench.add_argument("--bitplane", action="store_true", help="Force Bitplane engine for benchmark")
    parser_bench.add_argument("--reclassify", action="store_true", help="Copy files into easy/hard/hell categories")
    parser_bench.add_argument("--build", nargs=4, metavar=('TARGET', 'E', 'H', 'HELL'), help="Assemble dataset from categories")

    args = parser.parse_args()

    if args.command == "compress":
        if not os.path.exists(args.path):
            print(f"Error: file not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        if args.path.lower().endswith(".spx"):
            print(f"Error: input file is already a .spx archive: {args.path}", file=sys.stderr)
            sys.exit(1)
        from core.compress import compress_spx
        out_path = args.path.rsplit('.', 1)[0] + ".spx"
        
        preloaded_arr = None
        if args.path.lower().endswith(".raw"):
            if not args.raw_res:
                print("Error: --raw-res WxH is required for .raw files. Example: --raw-res 1024x768", file=sys.stderr)
                sys.exit(1)
            try:
                import numpy as np
                w_raw, h_raw = map(int, args.raw_res.lower().split('x'))
                c_raw = 3 if args.raw_mode == "RGB" else 1
                with open(args.path, 'rb') as f:
                    raw_bytes = f.read()
                    expected = w_raw * h_raw * c_raw
                    if len(raw_bytes) < expected:
                        print(f"Error: RAW file too small. Expected {expected} bytes, got {len(raw_bytes)}", file=sys.stderr)
                        sys.exit(1)
                    preloaded_arr = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape((h_raw, w_raw, c_raw))
            except Exception as e:
                print(f"Error loading RAW file: {e}", file=sys.stderr)
                sys.exit(1)

        # [v7.5.0] Logic: 
        # 1. Force Bitplane if --bitplane is set.
        # 2. Enable Auto-Selection if --optimize is set.
        # 3. Default to Standard (False) otherwise for deterministic CLI behavior.
        use_bitplane = False
        if args.bitplane:
            use_bitplane = True
        elif args.optimize:
            use_bitplane = None

        result = compress_spx(args.path if preloaded_arr is None else None, out_path, 
                               preloaded_arr=preloaded_arr, use_bitplane=use_bitplane)
        print(f"Compressed: {args.path} -> {out_path}")
        print(f"Size: {result.comp_size/1024:.2f} KB | Mode: {result.mode}")

    elif args.command == "decompress":
        if not os.path.exists(args.path):
            print(f"Error: file not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        from core.decompress import decompress_spx
        out_path = args.output
        if not out_path:
            out_path = args.path.rsplit('.', 1)[0] + "_restored.png"
        rec_arr, _ = decompress_spx(args.path, out_path)
        print(f"Decompressed: {args.path} -> {out_path}")
        print(f"Shape: {rec_arr.shape}")

    elif args.command == "benchmark":
        from core import test_suite
        # Relay correctly to test_suite.main()
        # [v6.2 Hub Sync] test_suite.main() uses positionals: codec, path
        original_argv = sys.argv
        sys.argv = [sys.argv[0], args.codec, args.path]
        if args.num_tests: sys.argv.extend(["-n", str(args.num_tests)])
        if args.workers: sys.argv.extend(["-w", str(args.workers)])
        if args.offset: sys.argv.extend(["--offset", str(args.offset)])
        if args.reclassify: sys.argv.append("--reclassify")
        if args.bitplane: sys.argv.append("--bitplane")
        if args.build: sys.argv.extend(["--build"] + list(args.build))
        try:
            test_suite.main()
        finally:
            sys.argv = original_argv

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
