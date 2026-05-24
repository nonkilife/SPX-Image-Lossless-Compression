# SPX Codec — Reviewer Instructions

**SPX (Space Express)** is a lossless image compression codec. This folder contains a self-contained Windows executable — no Python or any other installation required.

---

## Requirements

- Windows 10 / 11 (64-bit)
- No installation needed

---

## Usage

Open a terminal (Command Prompt or PowerShell) in this folder and run:

### Compress an image
```
spx.exe compress <image_path>
```
Accepts PNG, JPEG, BMP, TIFF, and RAW. Outputs a `.spx` file in the same folder as the input.

**Options:**
- `--optimize` — auto-selects the best internal coder per image
- `--bitplane` — forces the Bitplane rANS engine

**Example:**
```
spx.exe compress photo.png
spx.exe compress photo.png --optimize
```

### Decompress a .spx file
```
spx.exe decompress <file.spx>
```
Outputs a `<name>_restored.png` in the same folder by default.

**Options:**
- `--output <path>` — specify a custom output path

**Example:**
```
spx.exe decompress photo.spx
spx.exe decompress photo.spx --output result.png
```

### Show help
```
spx.exe --help
spx.exe compress --help
spx.exe decompress --help
```

---

## Notes

- SPX is **lossless**: the decompressed image is a pixel-perfect reconstruction of the original (MSE = 0).
- First launch may take 1–2 seconds as the exe extracts itself to a temp folder — this is normal PyInstaller behavior.
- The `.spx` format is SPX's own bitstream format and is not compatible with other tools.
