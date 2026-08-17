#![feature(portable_simd)]

use image::{GrayImage, ImageBuffer, Rgb, RgbImage};
use libblur::{
  gaussian_blur, BlurImage, BlurImageMut, ConvolutionMode, EdgeMode, EdgeMode2D, FastBlurChannels,
  GaussianBlurParams, ThreadingPolicy,
};
use napi::bindgen_prelude::*;
use napi::Result;
use napi_derive::napi;
use std::path::Path;
use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::SimdUint;
use std::simd::u8x32;

#[napi]
pub struct ImageProcessor {
  pub width: u32,
  pub height: u32,
  blur_buf_a: Vec<u8>,
  blur_buf_b: Vec<u8>,
  buf_phase: bool,
  has_previous: bool,
  diff_buffer: Vec<u8>,
  flood_fill_stack: Vec<(usize, usize)>,
  visited_bitset: Vec<u64>,
  dilate_buffer: Vec<u8>,
  morph_scratch: Vec<u8>,
}

#[napi]
pub type BoundingBox = [u32; 4]; // [x, y, x + w, y + h]

const MAX_RAW_BLOBS: usize = 1000;
const MAX_COVERAGE: f64 = 0.8;

struct Blob {
  min_x: u32,
  min_y: u32,
  max_x: u32, // exclusive
  max_y: u32, // exclusive
  area: u32,
}

fn boxes_touch_padded(a: &Blob, b: &Blob, pad: u32) -> bool {
  a.min_x <= b.max_x.saturating_add(pad)
    && b.min_x <= a.max_x.saturating_add(pad)
    && a.min_y <= b.max_y.saturating_add(pad)
    && b.min_y <= a.max_y.saturating_add(pad)
}

fn merge_into(target: &mut Blob, other: &Blob) {
  target.min_x = target.min_x.min(other.min_x);
  target.min_y = target.min_y.min(other.min_y);
  target.max_x = target.max_x.max(other.max_x);
  target.max_y = target.max_y.max(other.max_y);
  target.area += other.area;
}

fn cluster_blobs(blobs: Vec<Blob>, pad: u32, min_area: u32) -> Vec<Blob> {
  let mut clusters: Vec<Blob> = Vec::new();

  for blob in blobs {
    match clusters
      .iter_mut()
      .find(|cluster| boxes_touch_padded(cluster, &blob, pad))
    {
      Some(cluster) => merge_into(cluster, &blob),
      None => clusters.push(blob),
    }
  }

  // merging can bring previously separate clusters into range of each other
  loop {
    let mut merged_any = false;
    let mut i = 0;
    while i < clusters.len() {
      let mut j = i + 1;
      while j < clusters.len() {
        if boxes_touch_padded(&clusters[i], &clusters[j], pad) {
          let other = clusters.swap_remove(j);
          merge_into(&mut clusters[i], &other);
          merged_any = true;
        } else {
          j += 1;
        }
      }
      i += 1;
    }
    if !merged_any {
      break;
    }
  }

  clusters.retain(|cluster| cluster.area >= min_area);
  clusters.sort_by(|a, b| b.area.cmp(&a.area));
  clusters
}

#[napi]
impl ImageProcessor {
  #[napi(constructor)]
  pub fn new(width: u32, height: u32) -> Self {
    let num_pixels = (width * height) as usize;
    let bitset_size = num_pixels.div_ceil(64);
    ImageProcessor {
      width,
      height,
      blur_buf_a: vec![0u8; num_pixels],
      blur_buf_b: vec![0u8; num_pixels],
      buf_phase: false,
      has_previous: false,
      diff_buffer: vec![0u8; num_pixels],
      flood_fill_stack: Vec::with_capacity((width * height / 32) as usize),
      visited_bitset: vec![0u64; bitset_size],
      dilate_buffer: vec![0u8; num_pixels],
      morph_scratch: vec![0u8; num_pixels],
    }
  }

  #[napi]
  pub fn process_image(
    &mut self,
    frame: Uint8Array,
    threshold: u8,
    kernel_size: u8,
    dilate_size: u8,
    min_area: u32,
    update_reference: Option<bool>,
  ) -> Result<Vec<BoundingBox>> {
    let w = self.width as usize;
    let h = self.height as usize;

    {
      let current_buf = if self.buf_phase {
        &mut self.blur_buf_a
      } else {
        &mut self.blur_buf_b
      };
      Self::gaussian_blur_static(
        frame.as_ref(),
        current_buf.as_mut_slice(),
        w,
        h,
        kernel_size,
      )?;
    }

    if !self.has_previous {
      self.has_previous = true;
      self.buf_phase = !self.buf_phase;
      return Ok(vec![]);
    }

    let (previous_slice, current_slice) = if self.buf_phase {
      (self.blur_buf_b.as_slice(), self.blur_buf_a.as_slice())
    } else {
      (self.blur_buf_a.as_slice(), self.blur_buf_b.as_slice())
    };
    Self::detect_frame_differences(
      current_slice,
      previous_slice,
      &mut self.diff_buffer,
      threshold,
    );

    dilate_binary_box(
      &self.diff_buffer,
      &mut self.morph_scratch,
      &mut self.dilate_buffer,
      w,
      h,
      dilate_size as usize,
    );

    let blobs = find_blobs(
      &self.dilate_buffer,
      w,
      h,
      None,
      &mut self.flood_fill_stack,
      &mut self.visited_bitset,
    );

    if blobs.len() > MAX_RAW_BLOBS {
      self.buf_phase = !self.buf_phase;
      return Ok(vec![[0, 0, self.width, self.height]]);
    }

    let pad = (w.min(h) / 6) as u32;
    let clusters = cluster_blobs(blobs, pad, min_area);

    // clusters are pairwise disjoint after merging, the plain sum is exact
    let coverage: u64 = clusters
      .iter()
      .map(|c| ((c.max_x - c.min_x) as u64) * ((c.max_y - c.min_y) as u64))
      .sum();
    if coverage as f64 > MAX_COVERAGE * (w as f64) * (h as f64) {
      self.buf_phase = !self.buf_phase;
      return Ok(vec![[0, 0, self.width, self.height]]);
    }

    if update_reference.unwrap_or(true) {
      self.buf_phase = !self.buf_phase;
    }

    Ok(
      clusters
        .iter()
        .map(|c| [c.min_x, c.min_y, c.max_x, c.max_y])
        .collect(),
    )
  }

  #[napi]
  pub fn reconfigure(&mut self, width: u32, height: u32) {
    if self.width != width || self.height != height {
      self.width = width;
      self.height = height;
      self.has_previous = false;
      self.buf_phase = false;
      let num_pixels = (width * height) as usize;
      let bitset_size = num_pixels.div_ceil(64);
      self.blur_buf_a.resize(num_pixels, 0);
      self.blur_buf_b.resize(num_pixels, 0);
      self.diff_buffer.resize(num_pixels, 0);
      self.visited_bitset = vec![0u64; bitset_size];
      self.dilate_buffer.resize(num_pixels, 0);
      self.morph_scratch.resize(num_pixels, 0);
      self.flood_fill_stack.clear();
    }
  }

  #[napi]
  pub fn reset_state(&mut self) {
    self.has_previous = false;
    self.buf_phase = false;
  }

  #[napi]
  #[allow(clippy::too_many_arguments)]
  pub fn process_image_debug(
    &mut self,
    frame: Uint8Array,
    threshold: u8,
    kernel_size: u8,
    dilate_size: u8,
    min_area: u32,
    output_dir: String,
    prefix: String,
  ) -> Result<Vec<BoundingBox>> {
    let w = self.width as usize;
    let h = self.height as usize;

    // Step 1: Blur into current buffer
    {
      let current_buf = if self.buf_phase {
        &mut self.blur_buf_a
      } else {
        &mut self.blur_buf_b
      };
      Self::gaussian_blur_static(
        frame.as_ref(),
        current_buf.as_mut_slice(),
        w,
        h,
        kernel_size,
      )?;
    }

    // Save blurred image
    {
      let current_buf = if self.buf_phase {
        self.blur_buf_a.as_slice()
      } else {
        self.blur_buf_b.as_slice()
      };
      self.save_debug_image(
        current_buf,
        &output_dir,
        &format!("{}_1_blurred.png", prefix),
      )?;
    }

    if !self.has_previous {
      self.has_previous = true;
      self.buf_phase = !self.buf_phase;
      return Ok(vec![]);
    }

    // Step 2: Frame differences into diff_buffer
    {
      let (previous_slice, current_slice) = if self.buf_phase {
        (self.blur_buf_b.as_slice(), self.blur_buf_a.as_slice())
      } else {
        (self.blur_buf_a.as_slice(), self.blur_buf_b.as_slice())
      };
      Self::detect_frame_differences(
        current_slice,
        previous_slice,
        &mut self.diff_buffer,
        threshold,
      );
    }

    // Save difference image
    self.save_debug_image(
      &self.diff_buffer,
      &output_dir,
      &format!("{}_2_difference.png", prefix),
    )?;

    // Step 3: Dilate diff_buffer → dilate_buffer
    dilate_binary_box(
      &self.diff_buffer,
      &mut self.morph_scratch,
      &mut self.dilate_buffer,
      w,
      h,
      dilate_size as usize,
    );

    // Save dilated image
    self.save_debug_image(
      &self.dilate_buffer,
      &output_dir,
      &format!("{}_3_dilated.png", prefix),
    )?;

    // Step 4: Blobs on dilate_buffer, then cluster like process_image does
    let blobs = find_blobs(
      &self.dilate_buffer,
      w,
      h,
      None,
      &mut self.flood_fill_stack,
      &mut self.visited_bitset,
    );

    let bounding_boxes: Vec<BoundingBox> = if blobs.len() > MAX_RAW_BLOBS {
      vec![[0, 0, self.width, self.height]]
    } else {
      let pad = (w.min(h) / 6) as u32;
      cluster_blobs(blobs, pad, min_area)
        .iter()
        .map(|c| [c.min_x, c.min_y, c.max_x, c.max_y])
        .collect()
    };

    // Save bounding boxes image
    self.save_debug_image_with_boxes(
      &self.dilate_buffer,
      &bounding_boxes,
      &output_dir,
      &format!("{}_4_bounding_boxes.png", prefix),
    )?;

    self.buf_phase = !self.buf_phase;

    Ok(bounding_boxes)
  }

  fn gaussian_blur_static(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    kernel_size: u8,
  ) -> Result<()> {
    let src_image = BlurImage::borrow(input, width as u32, height as u32, FastBlurChannels::Plane);

    let mut dst_image =
      BlurImageMut::borrow(output, width as u32, height as u32, FastBlurChannels::Plane);

    let params = GaussianBlurParams {
      x_kernel: kernel_size as u32,
      x_sigma: 0.0,
      y_kernel: kernel_size as u32,
      y_sigma: 0.0,
    };

    gaussian_blur(
      &src_image,
      &mut dst_image,
      params,
      EdgeMode2D::new(EdgeMode::Clamp),
      ThreadingPolicy::Single,
      ConvolutionMode::FixedPoint,
    )
    .map_err(|e| napi::Error::from_reason(format!("Gaussian blur failed: {}", e)))
  }

  fn detect_frame_differences(
    current_frame: &[u8],
    previous_frame: &[u8],
    output: &mut [u8],
    threshold: u8,
  ) {
    let len = current_frame.len();
    let threshold_simd = u8x32::splat(threshold);
    let zero_simd = u8x32::splat(0);
    let max_simd = u8x32::splat(255);

    let chunks = len / 32;

    output[..chunks * 32]
      .chunks_exact_mut(32)
      .zip(current_frame[..chunks * 32].chunks_exact(32))
      .zip(previous_frame[..chunks * 32].chunks_exact(32))
      .for_each(|((out, current), previous)| {
        let c = u8x32::from_slice(current);
        let p = u8x32::from_slice(previous);

        let diff = c.saturating_sub(p) | p.saturating_sub(c);
        let mask = diff.simd_ge(threshold_simd);
        let result = mask.select(max_simd, zero_simd);

        result.copy_to_slice(out);
      });

    for i in (chunks * 32)..len {
      let diff = current_frame[i].abs_diff(previous_frame[i]);
      output[i] = if diff >= threshold { 255 } else { 0 };
    }
  }

  fn save_debug_image(&self, data: &[u8], output_dir: &str, filename: &str) -> Result<()> {
    let img = GrayImage::from_raw(self.width, self.height, data.to_vec())
      .ok_or_else(|| napi::Error::from_reason("Failed to create image"))?;

    let path = Path::new(output_dir).join(filename);
    img
      .save(&path)
      .map_err(|e| napi::Error::from_reason(format!("Failed to save image: {}", e)))?;

    Ok(())
  }

  fn save_debug_image_with_boxes(
    &self,
    data: &[u8],
    boxes: &[BoundingBox],
    output_dir: &str,
    filename: &str,
  ) -> Result<()> {
    let mut rgb_img: RgbImage = ImageBuffer::new(self.width, self.height);

    for (i, pixel) in data.iter().enumerate() {
      let x = (i % self.width as usize) as u32;
      let y = (i / self.width as usize) as u32;
      rgb_img.put_pixel(x, y, Rgb([*pixel, *pixel, *pixel]));
    }

    for bbox in boxes {
      let (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3]);

      for x in x1..x2 {
        if y1 < self.height {
          rgb_img.put_pixel(x, y1, Rgb([255, 0, 0]));
        }
        if y2 - 1 < self.height {
          rgb_img.put_pixel(x, y2 - 1, Rgb([255, 0, 0]));
        }
      }

      for y in y1..y2 {
        if x1 < self.width {
          rgb_img.put_pixel(x1, y, Rgb([255, 0, 0]));
        }
        if x2 - 1 < self.width {
          rgb_img.put_pixel(x2 - 1, y, Rgb([255, 0, 0]));
        }
      }
    }

    let path = Path::new(output_dir).join(filename);
    rgb_img
      .save(&path)
      .map_err(|e| napi::Error::from_reason(format!("Failed to save image: {}", e)))?;

    Ok(())
  }
}

fn dilate_binary_box(
  input: &[u8],
  scratch: &mut [u8],
  output: &mut [u8],
  width: usize,
  height: usize,
  r: usize,
) {
  if r == 0 {
    output.copy_from_slice(input);
    return;
  }

  let blocks_end = if width >= 32 + r { width - 32 - r } else { 0 };

  // Horizontal pass: input -> scratch (OR across columns within r).
  for y in 0..height {
    let row = y * width;
    let in_row = &input[row..row + width];
    let out_row = &mut scratch[row..row + width];

    let mut x = r;
    while x <= blocks_end {
      let base = x - r;
      let mut acc = u8x32::from_slice(&in_row[base..base + 32]);
      for k in 1..=2 * r {
        acc |= u8x32::from_slice(&in_row[base + k..base + k + 32]);
      }
      acc.copy_to_slice(&mut out_row[x..x + 32]);
      x += 32;
    }
    let simd_end = x;

    // Borders (and the whole row when the SIMD loop didn't run). `x` indexes
    // `out_row` while also slicing `in_row[lo..hi]`, so enumerate doesn't fit.
    #[allow(clippy::needless_range_loop)]
    for x in 0..width {
      if x >= r && x < simd_end {
        continue;
      }
      let lo = x.saturating_sub(r);
      let hi = (x + r + 1).min(width);
      out_row[x] = if in_row[lo..hi].iter().any(|&v| v != 0) {
        255
      } else {
        0
      };
    }
  }

  // Vertical pass: scratch -> output (OR across rows within r).
  for y in 0..height {
    let row = y * width;

    if y >= r && y + r < height {
      let top = (y - r) * width;
      let mut x = 0;
      while x + 32 <= width {
        let mut acc = u8x32::from_slice(&scratch[top + x..top + x + 32]);
        for k in 1..=2 * r {
          let o = top + k * width + x;
          acc |= u8x32::from_slice(&scratch[o..o + 32]);
        }
        acc.copy_to_slice(&mut output[row + x..row + x + 32]);
        x += 32;
      }
      for x in x..width {
        let mut v = 0u8;
        for k in 0..=2 * r {
          v |= scratch[top + k * width + x];
        }
        output[row + x] = v;
      }
    } else {
      // Border row: clamp the vertical window.
      let lo = y.saturating_sub(r);
      let hi = (y + r + 1).min(height);
      for x in 0..width {
        let mut v = 0u8;
        for yy in lo..hi {
          v |= scratch[yy * width + x];
        }
        output[row + x] = v;
      }
    }
  }
}

#[inline(always)]
fn get_bit(bitset: &[u64], idx: usize) -> bool {
  let word = idx / 64;
  let bit = idx % 64;
  (bitset[word] & (1u64 << bit)) != 0
}

#[inline(always)]
fn set_bit(bitset: &mut [u64], idx: usize) {
  let word = idx / 64;
  let bit = idx % 64;
  bitset[word] |= 1u64 << bit;
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn flood_fill(
  data: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  sy: usize,
  sx: usize,
  stack: &mut Vec<(usize, usize)>,
  visited: &mut [u64],
) -> (usize, usize, usize, usize, u32) {
  stack.clear();
  set_bit(visited, sy * width + sx);
  stack.push((sy, sx));

  let (mut min_x, mut min_y, mut max_x, mut max_y) = (sx, sy, sx, sy);
  let mut area: u32 = 0;

  while let Some((cy, cx)) = stack.pop() {
    min_x = min_x.min(cx);
    min_y = min_y.min(cy);
    max_x = max_x.max(cx);
    max_y = max_y.max(cy);
    area += 1;

    let row = cy * stride;
    if cx > 0 {
      let n = cy * width + cx - 1;
      if !get_bit(visited, n) && data[row + cx - 1] == 255 {
        set_bit(visited, n);
        stack.push((cy, cx - 1));
      }
    }
    if cx + 1 < width {
      let n = cy * width + cx + 1;
      if !get_bit(visited, n) && data[row + cx + 1] == 255 {
        set_bit(visited, n);
        stack.push((cy, cx + 1));
      }
    }
    if cy > 0 {
      let n = (cy - 1) * width + cx;
      if !get_bit(visited, n) && data[(cy - 1) * stride + cx] == 255 {
        set_bit(visited, n);
        stack.push((cy - 1, cx));
      }
    }
    if cy + 1 < height {
      let n = (cy + 1) * width + cx;
      if !get_bit(visited, n) && data[(cy + 1) * stride + cx] == 255 {
        set_bit(visited, n);
        stack.push((cy + 1, cx));
      }
    }
  }

  (min_x, min_y, max_x, max_y, area)
}

#[inline(always)]
fn find_blobs(
  data: &[u8],
  width: usize,
  height: usize,
  stride: Option<usize>,
  stack: &mut Vec<(usize, usize)>,
  visited: &mut [u64],
) -> Vec<Blob> {
  let stride = stride.unwrap_or(width);
  let mut blobs = Vec::new();

  visited.fill(0);

  let target = u8x32::splat(255);

  for y in 0..height {
    let row_offset = y * stride;
    let mut x = 0;

    while x + 32 <= width {
      let pixels = u8x32::from_slice(&data[row_offset + x..row_offset + x + 32]);
      let mask = pixels.simd_eq(target);

      if !mask.any() {
        x += 32;
        continue;
      }

      let mut mask_bits = mask.to_bitmask();
      while mask_bits != 0 {
        let offset = mask_bits.trailing_zeros() as usize;
        mask_bits &= mask_bits - 1;

        let cx = x + offset;
        if get_bit(visited, y * width + cx) {
          continue;
        }

        let (min_x, min_y, max_x, max_y, area) =
          flood_fill(data, width, height, stride, y, cx, stack, visited);

        blobs.push(Blob {
          min_x: min_x as u32,
          min_y: min_y as u32,
          max_x: (max_x + 1) as u32,
          max_y: (max_y + 1) as u32,
          area,
        });

        if blobs.len() > MAX_RAW_BLOBS {
          return blobs;
        }
      }

      x += 32;
    }

    while x < width {
      if !get_bit(visited, y * width + x) && data[row_offset + x] == 255 {
        let (min_x, min_y, max_x, max_y, area) =
          flood_fill(data, width, height, stride, y, x, stack, visited);

        blobs.push(Blob {
          min_x: min_x as u32,
          min_y: min_y as u32,
          max_x: (max_x + 1) as u32,
          max_y: (max_y + 1) as u32,
          area,
        });

        if blobs.len() > MAX_RAW_BLOBS {
          return blobs;
        }
      }
      x += 1;
    }
  }

  blobs
}
