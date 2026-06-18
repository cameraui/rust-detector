#![feature(portable_simd)]

use fast_morphology::{
  dilate, BorderMode, ImageSize, KernelShape, MorphScalar, MorphologyThreadingPolicy,
};
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
  temp_indices: Vec<usize>,
  visited_bitset: Vec<u64>,
  dilate_buffer: Vec<u8>,
}

#[napi]
pub type BoundingBox = [u32; 4]; // [x, y, x + w, y + h]

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
      temp_indices: Vec::with_capacity(32),
      visited_bitset: vec![0u64; bitset_size],
      dilate_buffer: vec![0u8; num_pixels],
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
      );
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

    let width = self.width;
    let height = self.height;
    Self::dilate_static(
      &self.diff_buffer,
      &mut self.dilate_buffer,
      width,
      height,
      dilate_size,
    );

    let bounding_boxes = _find_bounding_boxes(
      &self.dilate_buffer,
      w,
      h,
      min_area,
      None,
      &mut self.flood_fill_stack,
      &mut self.temp_indices,
      &mut self.visited_bitset,
    );

    self.buf_phase = !self.buf_phase;

    Ok(bounding_boxes)
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
      self.flood_fill_stack.clear();
      self.temp_indices.clear();
    }
  }

  #[napi]
  pub fn reset_state(&mut self) {
    self.has_previous = false;
    self.buf_phase = false;
  }

  #[napi]
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
      );
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
    {
      let width = self.width;
      let height = self.height;
      Self::dilate_static(
        &self.diff_buffer,
        &mut self.dilate_buffer,
        width,
        height,
        dilate_size,
      );
    }

    // Save dilated image
    self.save_debug_image(
      &self.dilate_buffer,
      &output_dir,
      &format!("{}_3_dilated.png", prefix),
    )?;

    // Step 4: Find bounding boxes on dilate_buffer directly
    let bounding_boxes = _find_bounding_boxes(
      &self.dilate_buffer,
      w,
      h,
      min_area,
      None,
      &mut self.flood_fill_stack,
      &mut self.temp_indices,
      &mut self.visited_bitset,
    );

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
  ) {
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
      EdgeMode2D::new(EdgeMode::Wrap),
      ThreadingPolicy::Adaptive,
      ConvolutionMode::FixedPoint,
    )
    .unwrap()
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

  fn dilate_static(input: &[u8], output: &mut [u8], width: u32, height: u32, size: u8) {
    let se_size = (size * 2 + 1) as usize;
    let kernel_shape = KernelShape::new(se_size, se_size);
    let structuring_element = vec![255u8; se_size * se_size];

    dilate(
      input,
      output,
      ImageSize::new(width as usize, height as usize),
      &structuring_element,
      kernel_shape,
      BorderMode::Wrap,
      MorphScalar::dup(0.0),
      MorphologyThreadingPolicy::default(),
    )
    .unwrap();
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
fn _find_bounding_boxes(
  data: &[u8],
  width: usize,
  height: usize,
  min_area: u32,
  stride: Option<usize>,
  stack: &mut Vec<(usize, usize)>,
  temp_indices: &mut Vec<usize>,
  visited: &mut [u64],
) -> Vec<BoundingBox> {
  let stride = stride.unwrap_or(width);
  let mut bounding_boxes = Vec::new();

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

      let mask_bits = mask.to_bitmask();
      temp_indices.clear();

      for offset in 0..32 {
        if (mask_bits & (1 << offset)) != 0 {
          let curr_x = x + offset;
          let visit_idx = y * width + curr_x;
          if !get_bit(visited, visit_idx) {
            temp_indices.push(visit_idx);
          }
        }
      }

      for &visit_idx in temp_indices.iter() {
        let cx = visit_idx % width;
        let cy = visit_idx / width;
        let data_idx = cy * stride + cx;

        if !get_bit(visited, visit_idx) && data[data_idx] == 255 {
          stack.clear();
          stack.push((cy, cx));

          let mut min_x = cx;
          let mut min_y = cy;
          let mut max_x = cx;
          let mut max_y = cy;
          let mut area = 0;

          while let Some((cy, cx)) = stack.pop() {
            let visit_idx = cy * width + cx;
            let data_idx = cy * stride + cx;

            if cy < height && cx < width && !get_bit(visited, visit_idx) && data[data_idx] == 255 {
              set_bit(visited, visit_idx);
              min_x = min_x.min(cx);
              min_y = min_y.min(cy);
              max_x = max_x.max(cx);
              max_y = max_y.max(cy);
              area += 1;

              if cx > 0 {
                stack.push((cy, cx - 1));
              }
              if cx + 1 < width {
                stack.push((cy, cx + 1));
              }
              if cy > 0 {
                stack.push((cy - 1, cx));
              }
              if cy + 1 < height {
                stack.push((cy + 1, cx));
              }
            }
          }

          if area >= min_area {
            bounding_boxes.push([
              min_x as u32,
              min_y as u32,
              (max_x + 1) as u32,
              (max_y + 1) as u32,
            ]);
          }
        }
      }

      x += 32;
    }

    while x < width {
      let idx = y * width + x;
      if !get_bit(visited, idx) && data[y * stride + x] == 255 {
        stack.clear();
        stack.push((y, x));

        let mut min_x = x;
        let mut min_y = y;
        let mut max_x = x;
        let mut max_y = y;
        let mut area = 0;

        while let Some((cy, cx)) = stack.pop() {
          let idx = cy * width + cx;
          if cy < height && cx < width && !get_bit(visited, idx) && data[cy * stride + cx] == 255 {
            set_bit(visited, idx);
            min_x = min_x.min(cx);
            min_y = min_y.min(cy);
            max_x = max_x.max(cx);
            max_y = max_y.max(cy);
            area += 1;

            if cx > 0 {
              stack.push((cy, cx - 1));
            }
            if cx + 1 < width {
              stack.push((cy, cx + 1));
            }
            if cy > 0 {
              stack.push((cy - 1, cx));
            }
            if cy + 1 < height {
              stack.push((cy + 1, cx));
            }
          }
        }

        if area >= min_area {
          bounding_boxes.push([
            min_x as u32,
            min_y as u32,
            (max_x + 1) as u32,
            (max_y + 1) as u32,
          ]);
        }
      }
      x += 1;
    }
  }

  bounding_boxes
}
