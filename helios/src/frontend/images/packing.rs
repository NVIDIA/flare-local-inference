//! Image-layout tensors consumed through prepared semantic bindings.
use super::ProcessedImages;
use crate::{
    Result,
    execution::Dtype,
    model::{ModelConfig, contract::ImageLayoutKind, inputs::UploadBatch},
    scheduler::RequestHandle,
};

pub(crate) fn tensor_layout(
    kind: ImageLayoutKind,
    name: &str,
    patches: usize,
) -> Result<(Dtype, usize)> {
    use Dtype::{Float32, Int32, Int64};
    let layout = match (kind, name) {
        (ImageLayoutKind::Qwen | ImageLayoutKind::Mimo, "vision_pixel_values") => {
            (Float32, patches * 1536)
        }
        (
            ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor | ImageLayoutKind::Moonvit | ImageLayoutKind::Step,
            "vision_pixel_values",
        ) => (Float32, patches * 588),
        (ImageLayoutKind::Bracketed, "vision_pixel_values") => (Float32, patches * 1176),
        (ImageLayoutKind::Radio, "vision_pixel_values") => (Float32, patches * 768),
        (ImageLayoutKind::Glm, "vision_pixel_values") => (Float32, patches * 1176),
        (_, "vision_position_ids") => (Int64, patches * 2),
        (ImageLayoutKind::Step, "vision_cu_seqlens") => (Int32, super::step::MAX_VIEWS + 1),
        (_, "vision_cu_seqlens") => (Int32, 2),
        (_, "vision_num_patches" | "vision_feature_offset") => (Int32, 1),
        (
            ImageLayoutKind::Bracketed | ImageLayoutKind::Mimo
            | ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio,
            "vision_num_features",
        ) => (Int32, 1),
        (
            ImageLayoutKind::Qwen
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio,
            "vision_interpolation_indices",
        ) => (Int64, patches * 4),
        (
            ImageLayoutKind::Qwen
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio,
            "vision_interpolation_weights",
        ) => (Float32, patches * 4),
        (ImageLayoutKind::Mimo, "vision_column_indices" | "vision_row_indices") => {
            (Int32, patches / 4)
        }
        (
            ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor | ImageLayoutKind::Moonvit | ImageLayoutKind::Radio,
            "vision_merge_indices",
        ) => (Int32, patches),
        (ImageLayoutKind::Step, "vision_downsample1_indices") => (Int32, patches / 4 * 9),
        (ImageLayoutKind::Step, "vision_downsample2_indices") => (Int32, patches / 16 * 9),
        _ => return Err(format!("image layout has no tensor {name}").into()),
    };
    Ok(layout)
}

pub(crate) fn prepare(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images.geometry.get(image).ok_or("missing scheduled image")?;
    if destination_feature_offset.checked_add(geometry.tokens)
        .is_none_or(|end| end > config.contract.vision.features_per_request) {
        return Err("invalid destination vision feature extent".into());
    }
    i32::try_from(destination_feature_offset).map_err(|_| "vision feature offset exceeds int32")?;
    match config.contract.frontend.image_layout
        .ok_or("image layout is not prepared")? {
        ImageLayoutKind::Bracketed => bracketed(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Qwen | ImageLayoutKind::Glm => qwen(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Mimo => mimo(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor => deepseek(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Moonvit | ImageLayoutKind::Radio => spatial_merge(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Step => step(config, handle, images, image, destination_feature_offset),
    }
}

fn qwen(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let pixel_width = if matches!(
        config.contract.frontend.image_layout,
        Some(ImageLayoutKind::Moonvit | ImageLayoutKind::Step)
    ) {
        588
    } else if config.contract.frontend.image_layout == Some(ImageLayoutKind::Radio) {
        768
    } else if config.contract.frontend.image_layout == Some(ImageLayoutKind::Glm) {
        1176
    } else {
        1536
    };
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image]
        .iter()
        .map(|geometry| geometry.patches)
        .sum();
    let end = start + geometry.patches;
    if geometry.patches == 0
        || geometry.patches
            > *config
                .contract
                .vision
                .patch_capacities
                .last()
                .ok_or("missing vision capacities")?
        || end / config.contract.vision.patches_per_feature
            > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid vision graph input extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * pixel_width..end * pixel_width)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    if config.contract.frontend.image_layout != Some(ImageLayoutKind::Glm) {
        staging.i64(
            "vision_interpolation_indices",
            0,
            images
                .interpolation_indices
                .get(start..end)
                .ok_or("truncated vision interpolation indices")?
                .iter()
                .flatten()
                .copied(),
        );
        staging.put(
            "vision_interpolation_weights",
            0,
            images
                .interpolation_weights
                .get(start..end)
                .ok_or("truncated vision interpolation weights")?
                .iter()
                .flatten()
                .copied()
                .flat_map(f32::to_le_bytes),
        );
    }
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    if config.contract.frontend.image_layout != Some(ImageLayoutKind::Step) {
        staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    }
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32(
        "vision_feature_offset",
        0,
        [destination_feature_offset as i32],
    );
    Ok(staging)
}

fn mimo(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    if h == 0
        || w == 0
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || geometry.patches > 16384
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
        || end / 4 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid MiMo vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 1536..end * 1536)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let (columns, rows) = column_permutation(h / 2, w / 2);
    staging.i32("vision_column_indices", 0, columns);
    staging.i32("vision_row_indices", 0, rows);
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}
fn column_permutation(height: usize, width: usize) -> (Vec<i32>, Vec<i32>) {
    let columns: Vec<_> = (0..width)
        .flat_map(|x| (0..height).map(move |y| (y * width + x) as i32))
        .collect();
    let mut rows = vec![0; columns.len()];
    for (index, column) in columns.iter().enumerate() {
        rows[*column as usize] = index as i32;
    }
    (columns, rows)
}

fn deepseek(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    let capacity = *config.contract.vision.patch_capacities.last().ok_or("missing vision capacities")?;
    let actual = h
        .checked_mul(w)
        .filter(|n| *n > 0 && *n <= capacity)
        .ok_or("invalid vision grid")?;
    if geometry.patches > capacity
        || geometry.patches != h.div_ceil(3) * w.div_ceil(3) * 9
        || geometry.tokens != geometry.patches / 9
        || end / 9 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid DeepSeek vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 588..end * 588)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let mut indices = Vec::with_capacity(geometry.patches);
    for row in 0..h.div_ceil(3) {
        for col in 0..w.div_ceil(3) {
            for dy in 0..3 {
                for dx in 0..3 {
                    let (y, x) = (row * 3 + dy, col * 3 + dx);
                    indices.push(if y < h && x < w {
                        (y * w + x) as i32
                    } else {
                        -1
                    });
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_cu_seqlens", 0, [0, actual as i32]);
    staging.i32("vision_num_patches", 0, [actual as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}

fn spatial_merge(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    let max_side = if config.contract.frontend.image_layout == Some(ImageLayoutKind::Radio) {
        13312
    } else {
        512
    };
    if h == 0
        || w == 0
        || h > max_side
        || w > max_side
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
    {
        return Err("invalid spatial merge vision grid".into());
    }
    let mut staging = qwen(config, handle, images, image, destination_feature_offset)?;
    let mut indices = Vec::with_capacity(geometry.patches);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    indices.push(((y + dy) * w + x + dx) as i32);
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    Ok(staging)
}

fn convolution_indices(side: usize, base: usize) -> Vec<i32> {
    let mut indices = Vec::with_capacity(side * side / 4 * 9);
    for y in (0..side).step_by(2) {
        for x in (0..side).step_by(2) {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (y, x) = (y as isize + dy, x as isize + dx);
                    indices.push(
                        if y < 0 || x < 0 || y >= side as isize || x >= side as isize {
                            -1
                        } else {
                            (base + y as usize * side + x as usize) as i32
                        },
                    );
                }
            }
        }
    }
    indices
}

fn step(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let ny = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid crop rows")?;
    let nx = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid crop columns")?;
    let local = ny
        .checked_mul(nx)
        .filter(|n| *n < super::step::MAX_VIEWS)
        .ok_or("too many image crops")?;
    if geometry.grid_thw[0] != 1
        || (nx == 0) != (ny == 0)
        || geometry.patches != local * 1296 + 2704
        || geometry.tokens != geometry.patches / 16
    {
        return Err("invalid Step image geometry".into());
    }
    let mut staging = qwen(config, handle, images, image, destination_feature_offset)?;
    let mut cu = vec![0];
    let (mut first, mut second) = (vec![], vec![]);
    let mut base = 0;
    for view in 0..=local {
        let side = if view == local { 52 } else { 36 };
        first.extend(convolution_indices(side, base));
        second.extend(convolution_indices(side / 2, base / 4));
        base += side * side;
        cu.push(base as i32);
    }
    cu.resize(super::step::MAX_VIEWS + 1, base as i32);
    staging.i32("vision_cu_seqlens", 0, cu);
    staging.i32("vision_downsample1_indices", 0, first);
    staging.i32("vision_downsample2_indices", 0, second);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    Ok(staging)
}

fn bracketed(config: &ModelConfig, handle: RequestHandle, images: &ProcessedImages, image: usize, destination_feature_offset: usize) -> Result<UploadBatch> {
    let geometry = images.geometry.get(image).ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let [t, h, w] = geometry.grid_thw;
    if t != 1 || h <= 0 || w <= 0 || h%2 != 0 || w%2 != 0
        || (h as usize).checked_mul(w as usize) != Some(geometry.patches)
        || geometry.tokens != geometry.patches/4 || config.contract.vision.patches_per_feature != 4
        || geometry.patches > *config.contract.vision.patch_capacities.last().ok_or("missing vision capacities")?
        || handle.slot as usize >= config.limits.max_active || handle.generation == 0 || handle.generation > i64::MAX as u64 {
        return Err("invalid temporal patch-merge image extent".into());
    }
    let mut staging = UploadBatch::default();
    staging.put("vision_pixel_values", 0, images.pixel_values.get(start*1176..end*1176).ok_or("truncated vision pixels")?.iter().copied().flat_map(f32::to_le_bytes));
    staging.i64("vision_position_ids", 0, images.position_ids.get(start..end).ok_or("truncated vision positions")?.iter().flatten().copied());
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}
