//! MiMo image preprocessing profile.

use super::*;

pub(super) fn load(
    settings: &MimoImageSettings,
    limits: ImageLimits,
) -> Result<patches::PatchProcessor> {
    if settings.min_pixels == 0
        || settings.max_pixels < settings.min_pixels
        || settings.image_mean.iter().any(|v| !v.is_finite())
        || settings
            .image_std
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("invalid prepared MiMo image settings".into());
    }
    Ok(patches::PatchProcessor {
        limits,
        patch_size: PATCH,
        min_pixels: settings.min_pixels,
        max_pixels: settings.max_pixels,
        resize: resize::pillow,
        normalization: patches::Normalization::Channel {
            mean: settings.image_mean,
            std: settings.image_std,
        },
    })
}
