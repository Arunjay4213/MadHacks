// Optical post pass. At rest it is clean (gentle barrel, vignette, faint grain only).
// A commit adds a refracting shockwave with 2-3 px chromatic aberration on its front;
// a localized edit also gets a Sobel edge shimmer near its centre for the first 250 ms.
export const postFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uSource;   // 1x1 placeholder when there is no external source
uniform float uSourceMix;    // 0 procedural .. 1 external
uniform float uSourceAspect; // external width / height
uniform vec2 uRes;
uniform float uTime;
uniform float uImpulseAge;   // seconds since the commit impulse; large when none
uniform vec2 uImpulseCenter; // 0..1, GL orientation
uniform float uImpulseLocal; // 1 for a localized edit
uniform float uReduced;

vec2 coverUv(vec2 uv) {
  float a = (uRes.x / uRes.y) / uSourceAspect;
  return a > 1.0 ? vec2(uv.x, (uv.y - 0.5) / a + 0.5) : vec2((uv.x - 0.5) * a + 0.5, uv.y);
}
vec3 inputAt(vec2 uv) {
  vec3 w = texture2D(uScene, uv).rgb;
  if (uSourceMix < 0.001) return w;
  return mix(w, texture2D(uSource, coverUv(uv)).rgb, uSourceMix);
}
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec2 q = vUv - 0.5;
  float r2 = dot(q, q);
  vec2 uv = 0.5 + q * (1.0 - 0.03 * r2);  // gentle barrel
  vec3 col;
  bool pulseActive = uImpulseAge >= 0.0 && uImpulseAge < 1.0 && uReduced < 0.5;
  if (!pulseActive) {
    col = inputAt(uv);
  } else {
    vec2 aspect = vec2(uRes.x / uRes.y, 1.0);
    vec2 d = (vUv - uImpulseCenter) * aspect;
    float dist = length(d);
    vec2 dir = dist > 0.0001 ? d / dist : vec2(0.0);
    float radius = uImpulseAge * 1.25;
    float wave = exp(-pow((dist - radius) * 16.0, 2.0)) * (1.0 - uImpulseAge);
    uv -= dir * wave * 0.028 / aspect;
    vec2 ca = dir * wave * 2.5 / uRes;  // at most about 2.5 px, only on the wave front
    col = vec3(inputAt(uv + ca).r, inputAt(uv).g, inputAt(uv - ca).b);
    // Localized edit: Sobel edge shimmer near the centre, first 250 ms only.
    float shimmer = uImpulseLocal * (1.0 - smoothstep(0.0, 0.25, uImpulseAge)) * smoothstep(0.45, 0.0, dist);
    if (shimmer > 0.001) {
      vec2 t = 1.5 / uRes;
      float tl = luma(inputAt(uv + vec2(-t.x, t.y))), tc = luma(inputAt(uv + vec2(0.0, t.y))), tr = luma(inputAt(uv + t));
      float ml = luma(inputAt(uv - vec2(t.x, 0.0))), mr = luma(inputAt(uv + vec2(t.x, 0.0)));
      float bl = luma(inputAt(uv - t)), bc = luma(inputAt(uv - vec2(0.0, t.y))), br = luma(inputAt(uv + vec2(t.x, -t.y)));
      float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
      float gy = (tl + 2.0 * tc + tr) - (bl + 2.0 * bc + br);
      col += vec3(0.62, 0.78, 1.0) * smoothstep(0.08, 0.5, length(vec2(gx, gy))) * shimmer * 0.5;
    }
  }
  col *= 1.0 - smoothstep(0.2, 0.75, r2 * 1.6) * 0.42;  // vignette
  col += (hash(vUv * uRes + fract(uTime) * 61.0) - 0.5) * 0.018;  // grain
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;
