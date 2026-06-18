# @camera.ui/rust-detector

Native Rust motion/object detector for the [camera.ui](https://github.com/seydx/camera.ui) ecosystem.

Built with [napi-rs](https://napi.rs) — ships prebuilt binaries for Linux (glibc/musl), macOS, Windows and FreeBSD across x64, arm64 and riscv64, so there is no compile step on install.

## Installation

```bash
npm install @camera.ui/rust-detector
```

## Usage

```ts
import { ImageProcessor } from '@camera.ui/rust-detector';

const processor = new ImageProcessor(width, height);

// frame: grayscale Uint8Array of length width * height
const boxes = processor.processImage(frame, threshold, kernelSize, dilateSize, minArea);
// boxes: Array<[x, y, w, h]>
```

`reconfigure(width, height)` resizes the processor and `resetState()` clears the
internal frame state between streams.

## Development

```bash
npm install
npm run build        # release build (napi build --platform --release)
npm run build:debug  # debug build
npm run lint         # cargo clippy + eslint
```

---

_Part of the camera.ui ecosystem - A comprehensive camera management solution._
