// Luminous filament ring: points displaced by 3D simplex curl noise, driven by audio bands.
import { simplex3d } from './noise';

export const singularityVert = /* glsl */ `
attribute float aAngle;
attribute float aSeed;
attribute float aLane;
uniform float uTime;
uniform float uLow;
uniform float uVocal;
uniform float uHigh;
uniform float uScale;    // 1 at rest; implodes toward 0.6 on a commit
uniform float uBurst;    // 1 at the start of the burst, fading to 0
uniform float uPixelRatio;
varying vec3 vColor;
varying float vAlpha;

${simplex3d}

void main() {
  float a = aAngle;
  float radius = (1.0 + (aLane - 0.5) * 0.16 + uLow * 0.22) * uScale;
  vec3 p = vec3(cos(a) * radius, 0.0, sin(a) * radius);
  // Tilt the ring toward the viewer so it reads as a halo.
  float ct = cos(0.5), st = sin(0.5);  // about 0.48 height/width ellipse
  p = vec3(p.x, p.y * ct - p.z * st, p.y * st + p.z * ct);
  vec3 flow = curlNoise(p * 1.4 + vec3(0.0, uTime * 0.22, aSeed * 0.35));
  p += flow * (0.035 + uVocal * 0.32) * (0.35 + aSeed);
  p += (vec3(fract(aSeed * 91.7), fract(aSeed * 53.3), fract(aSeed * 27.1)) - 0.5) * uHigh * 0.06;
  // Burst: sparks fly outward, faster seeds further.
  p += normalize(p + 1e-4) * uBurst * (0.25 + aSeed * 1.1);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float energy = clamp(uLow * 0.4 + uVocal * 0.8 + uHigh * 0.5, 0.0, 1.5);
  gl_PointSize = (1.4 + aSeed * 2.4 + uHigh * 2.5 + uBurst * 3.0) * uPixelRatio * (3.2 / -mv.z);
  vec3 cyan = vec3(0.42, 0.86, 1.0);
  vec3 violet = vec3(0.70, 0.48, 1.0);
  vec3 warm = vec3(1.0, 0.92, 0.80);
  vColor = mix(mix(cyan, violet, 0.5 + 0.5 * sin(a * 3.0 + uTime * 0.4)), warm, clamp(uVocal * 0.6 + uBurst * 0.5, 0.0, 0.8));
  vAlpha = (0.22 + 0.55 * energy + uBurst * 0.6) * (0.5 + 0.5 * aSeed);
}
`;

export const singularityFrag = /* glsl */ `
precision highp float;
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = pow(smoothstep(0.5, 0.0, d), 1.6) * vAlpha;
  if (a < 0.003) discard;
  gl_FragColor = vec4(vColor, a);  // AdditiveBlending applies the alpha
}
`;
