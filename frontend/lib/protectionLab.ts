/**
 * Model-free image perturbation baselines for the protection lab.
 *
 * These transforms are deliberately described as proxies: they do not
 * implement Unlearnable Examples or PhotoGuard's model-gradient objectives.
 */

function copyImage(src: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}

function clamp(value: number): number {
  return Math.max(0, Math.min(255, value));
}

/** Small deterministic multi-frequency pattern used as a poisoning proxy. */
export function applyTrainingInterferenceProxy(src: ImageData, strength = 3): ImageData {
  const out = copyImage(src);
  const { width, height } = src;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const pattern = Math.sin(x * 1.71 + y * 0.37) * Math.cos(y * 1.29 - x * 0.23);
      for (let c = 0; c < 3; c++) {
        const channelPhase = c * 1.7;
        const value = pattern * Math.sin((x + y) * 0.41 + channelPhase) * strength;
        out.data[i + c] = clamp(src.data[i + c] + value);
      }
    }
  }
  return out;
}

/** Multi-band high-frequency pattern used as a VAE/editing proxy. */
export function applyGenerationInterferenceProxy(src: ImageData, strength = 3): ImageData {
  const out = copyImage(src);
  const { width, height } = src;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const waveA = Math.sin(x * 2.13 + y * 0.71);
      const waveB = Math.cos(y * 1.87 - x * 0.53);
      const waveC = Math.sin((x + y) * 2.77);
      for (let c = 0; c < 3; c++) {
        const phase = c * 0.9;
        const value = (waveA + waveB + waveC * Math.cos(phase)) * (strength / 3);
        out.data[i + c] = clamp(src.data[i + c] + value);
      }
    }
  }
  return out;
}

/** Visible attribution mark. It helps identify an export but does not block editing. */
export function addAttributionWatermark(src: ImageData): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = src.width;
  canvas.height = src.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return copyImage(src);
  ctx.putImageData(src, 0, 0);
  const label = "AI EDITING TEST · PROXY";
  const fontSize = Math.max(11, Math.round(src.width * 0.025));
  ctx.font = `600 ${fontSize}px Arial, sans-serif`;
  const pad = Math.round(fontSize * 0.7);
  const textWidth = ctx.measureText(label).width;
  const x = Math.max(pad, src.width - textWidth - pad * 2);
  const y = Math.max(fontSize + pad, src.height - pad);
  ctx.fillStyle = "rgba(0, 0, 0, 0.48)";
  ctx.fillRect(x - pad, y - fontSize - pad / 2, textWidth + pad * 2, fontSize + pad);
  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.fillText(label, x, y);
  return ctx.getImageData(0, 0, src.width, src.height);
}
