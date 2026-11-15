// Procedural alpine valley: a painted, layered 2.5D simulation, not GPU model output.
import { noise2d } from './noise';

export const fullscreenVert = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const worldFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform vec2 uRes;
uniform vec2 uTilt;      // -1..1, eased pointer
uniform float uWinter;   // 0..1.3
uniform float uRiver;
uniform float uNight;
uniform float uSunset;
uniform float uRain;
uniform float uFog;
uniform float uRiverX;   // river centre offset, -0.5..0.5

${noise2d}

const float HORIZON = 0.38;
const int LAYERS = 5;

struct Light { vec3 sky0; vec3 sky1; vec3 sun; vec2 sunPos; vec3 haze; float amb; };

Light lighting() {
  Light L;
  // Muted, filmic palette: dusty blue day, ember dusk, ink-blue night.
  vec3 dayTop = vec3(0.34, 0.44, 0.56), dayHor = vec3(0.72, 0.75, 0.76);
  vec3 setTop = vec3(0.24, 0.21, 0.32), setHor = vec3(0.86, 0.56, 0.38);
  vec3 nitTop = vec3(0.010, 0.016, 0.040), nitHor = vec3(0.055, 0.075, 0.12);
  L.sky0 = mix(mix(dayTop, setTop, uSunset), nitTop, uNight);
  L.sky1 = mix(mix(dayHor, setHor, uSunset), nitHor, uNight);
  L.sun = mix(mix(vec3(1.0, 0.93, 0.82), vec3(1.0, 0.58, 0.32), uSunset), vec3(0.50, 0.58, 0.80) * 0.45, uNight);
  L.sunPos = mix(vec2(0.70, 0.78 - 0.32 * uSunset), vec2(0.28, 0.78), uNight);
  L.haze = mix(L.sky1, vec3(0.66, 0.68, 0.70), 0.4);
  L.amb = mix(1.0, 0.22, uNight);
  float grey = clamp(uRain * 0.7 + uFog * 0.5, 0.0, 0.9);  // overcast flattens toward grey
  vec3 g0 = mix(vec3(0.36, 0.39, 0.42), vec3(0.04, 0.05, 0.07), uNight);
  vec3 g1 = mix(vec3(0.60, 0.62, 0.63), vec3(0.08, 0.09, 0.11), uNight);
  L.sky0 = mix(L.sky0, g0, grey);
  L.sky1 = mix(L.sky1, g1, grey);
  L.haze = mix(L.haze, g1, grey);
  L.sun *= 1.0 - grey * 0.75;
  return L;
}

// Camera drift and pointer parallax: nearer layers move more.
vec2 parallax(vec2 uv, float depth) {
  float k = mix(0.004, 0.045, depth);
  vec2 drift = vec2(sin(uTime * 0.03) * 0.6, sin(uTime * 0.021) * 0.25);
  return uv + (uTilt * vec2(1.0, 0.5) + drift) * k;
}

float ridgeHeight(float x, int i) {
  float fi = float(i);
  float depth = fi / float(LAYERS - 1);
  float freq = mix(1.6, 3.6, depth);
  float base = mix(HORIZON + 0.20, HORIZON - 0.02, depth);
  float amp = mix(0.30, 0.14, depth);
  return base + amp * (ridge1(x * freq, fi * 3.7) - 0.35);
}

// 1 inside a pine silhouette standing on a ground line. x in scene units, y above the ground (screen units).
float pines(float x, float y, float density, float seed, float hMax, float clusterCut) {
  if (y < 0.0 || y > hMax) return 0.0;
  float cell = x * density;
  float ci = floor(cell);
  for (int k = -1; k <= 1; k++) {
    float cc = ci + float(k);
    float r = hash11(cc * 1.37 + seed);
    float cluster = vnoise1(cc * 0.18 + seed);
    if (cluster < clusterCut || r < 0.22) continue;
    float H = hMax * (0.45 + 0.55 * r);
    float t = y / H;  // 0 at the base, 1 at the tip
    if (t > 1.0) continue;
    float tx = cell - cc - 0.5 + (hash11(cc + seed * 3.0) - 0.5) * 0.5;
    tx += sin(uTime * 0.7 + cc) * 0.03 * t;
    float tier = fract(t * 5.0 + r);
    float halfW = max((1.0 - t) * 0.42 * (0.68 + 0.32 * (1.0 - tier)), 0.035);
    if (abs(tx) < halfW) return 1.0;
  }
  return 0.0;
}

vec3 skyColor(vec2 uv, vec2 p, Light L) {
  float h = clamp((uv.y - HORIZON) / (1.0 - HORIZON), 0.0, 1.0);
  vec3 c = mix(L.sky1, L.sky0, pow(h, 0.7));
  vec2 d = (uv - L.sunPos) * vec2(uRes.x / uRes.y, 1.0);
  float r = length(d);
  // Soft sun: small disc, gentle bloom. No starburst.
  c += L.sun * (smoothstep(0.022, 0.018, r) * 1.2 + exp(-r * 9.0) * 0.30 + exp(-r * 2.5) * 0.07);
  vec2 sp = p * 140.0;
  float s = hash12(floor(sp));
  float tw = 0.6 + 0.4 * sin(uTime * 2.0 + s * 40.0);
  c += uNight * step(0.9968, s) * smoothstep(0.5, 0.0, length(fract(sp) - 0.5)) * tw * h * (1.0 - uFog) * 0.8;
  vec2 cp = vec2(p.x * 1.3 + uTime * 0.012, (uv.y - HORIZON) * 5.0);
  float n = fbm(cp + fbm(cp * 0.7 + uTime * 0.01) * 1.4);
  float cover = mix(0.56, 0.30, clamp(uRain + uFog * 0.5, 0.0, 1.0));
  float cl = smoothstep(cover, cover + 0.32, n) * smoothstep(0.0, 0.14, uv.y - HORIZON);
  vec3 cloudLit = mix(L.sky1 * 1.02, L.sun * 0.75 + L.sky1 * 0.45, 0.4 * exp(-length(uv - L.sunPos) * 2.2));
  return mix(c, cloudLit * L.amb + vec3(0.015), cl * 0.8);
}

// Rock faces lit in 2D: a noise field tilts the surface normal, stretched vertically into couloirs.
vec3 rockShade(float x, float y, float h, int i, float depth, Light L) {
  float fi = float(i);
  float e = 0.002;
  float slope = (ridgeHeight(x + e, i) - ridgeHeight(x - e, i)) / (2.0 * e);
  float below = h - y;
  vec2 rp = vec2(x * mix(5.0, 9.0, depth), y * mix(9.0, 15.0, depth)) + fi * 11.0;
  float n0 = fbm(rp);
  float nx = fbm(rp + vec2(0.04, 0.0));
  float ny = fbm(rp + vec2(0.0, 0.04));
  vec3 nrm = normalize(vec3((n0 - nx) * 9.0 - slope * 0.45, (n0 - ny) * 5.0 + 0.55, 1.0));
  vec3 sunDir = normalize(vec3((L.sunPos.x - 0.5) * 2.0, 0.6, 0.55));
  float lit = clamp(dot(nrm, sunDir), 0.0, 1.0);
  float ao = mix(0.5, 1.0, exp(-below * 5.0));  // crests catch light, lower slopes fall into shade
  vec3 rock = mix(vec3(0.16, 0.17, 0.18), vec3(0.35, 0.35, 0.34), n0);
  vec3 grass = mix(vec3(0.16, 0.19, 0.14), vec3(0.23, 0.26, 0.19), n0);
  vec3 ground = mix(rock, grass, smoothstep(0.06, 0.22, below) * depth);
  // Permanent snow only on the far peaks, broken by gullies; winter covers every slope.
  float capZone = (1.0 - smoothstep(0.2, 0.55, depth)) * smoothstep(0.09 + 0.05 * n0, 0.0, below);
  float snow = capZone * smoothstep(0.42, 0.62, n0 + nrm.x * 0.3);
  snow = max(snow, clamp(uWinter, 0.0, 1.0) * smoothstep(0.3, 0.7, n0 + lit * 0.4 + 0.25));
  vec3 snowCol = mix(vec3(0.52, 0.57, 0.65), vec3(0.92, 0.93, 0.94), lit);
  ground = mix(ground, snowCol, clamp(snow, 0.0, 1.0));
  vec3 lightCol = mix(L.sky1 * 0.42, L.sun, lit) * L.amb + L.sky0 * 0.12;
  return ground * lightCol * ao;
}

vec3 pineColor(float t, float x, Light L) {
  vec3 c = mix(vec3(0.035, 0.055, 0.050), vec3(0.07, 0.09, 0.075), hash11(floor(x * 40.0)));
  c *= (L.sky1 * 0.5 + L.sun * 0.25) * L.amb + 0.02;
  // Snow dusting on the tiers in winter.
  float dust = clamp(uWinter, 0.0, 1.0) * step(0.55, fract(t * 90.0 + vnoise(vec2(x * 60.0, t * 30.0))));
  return mix(c, vec3(0.80, 0.83, 0.88) * L.amb * 0.85, dust * 0.5);
}

// Sky, mountains and ridge pines. Used for the main view and the river reflection.
vec3 backdrop(vec2 uv0, Light L, float aspect) {
  vec2 uv = parallax(uv0, 0.0);
  vec2 p = vec2((uv.x - 0.5) * aspect, uv.y);
  vec3 col = skyColor(uv, p, L);
  float hs[LAYERS];
  float ys[LAYERS];
  float xs[LAYERS];
  int hit = -1;
  bool tree = false;
  // Nearest covering layer wins; ridge pines stand above the two nearest crests.
  for (int i = 0; i < LAYERS; i++) {
    float depth = float(i) / float(LAYERS - 1);
    vec2 u = parallax(uv0, 0.15 + depth * 0.6);
    float x = (u.x - 0.5) * aspect;
    float h = ridgeHeight(x, i);
    hs[i] = h; ys[i] = u.y; xs[i] = x;
    if (u.y < h) { hit = i; tree = false; }
    else if (i >= LAYERS - 2 && pines(x, u.y - h + 0.006, mix(30.0, 22.0, depth), float(i) * 7.0, mix(0.035, 0.07, depth), 0.42) > 0.5) {
      hit = i; tree = true;
    }
  }
  if (hit >= 0) {
    float depth = float(hit) / float(LAYERS - 1);
    vec3 c = tree ? pineColor(ys[hit] - hs[hit], xs[hit], L) : rockShade(xs[hit], ys[hit], hs[hit], hit, depth, L);
    col = mix(c, L.haze * L.amb, clamp((1.0 - depth) * 0.78 + uFog * 0.4, 0.0, 0.93));
  }
  // Mist resting just above each nearer crest, in front of everything farther away.
  for (int i = 0; i < LAYERS; i++) {
    if (i <= hit) continue;
    float depth = float(i) / float(LAYERS - 1);
    float above = ys[i] - hs[i];
    float mist = exp(-max(above, 0.0) * 14.0) * (0.35 + 0.65 * fbm(vec2(xs[i] * 2.0 - uTime * 0.02 * (1.0 + depth), ys[i] * 6.0)));
    col = mix(col, L.haze * L.amb, mist * (0.32 + uFog * 0.5) * (1.0 - depth * 0.35));
  }
  return col;
}

void main() {
  float aspect = uRes.x / uRes.y;
  Light L = lighting();
  vec2 uv = vUv;
  vec3 col = backdrop(uv, L, aspect);
  float floorY = ridgeHeight((parallax(uv, 0.75).x - 0.5) * aspect, LAYERS - 1) - 0.02;
  // Valley floor and river, in perspective below the nearest ridge.
  if (uv.y < floorY) {
    vec2 u = parallax(uv, 1.0);
    float z = 0.06 / max(floorY - u.y + 0.015, 0.001);  // distance
    vec2 g = vec2((u.x - 0.5) * aspect * z * 3.0, z);
    float tex = fbm(g * vec2(2.0, 1.2));
    vec3 grass = mix(vec3(0.17, 0.21, 0.13), vec3(0.27, 0.30, 0.20), tex);
    vec3 snowG = mix(vec3(0.68, 0.72, 0.78), vec3(0.90, 0.91, 0.93), tex);
    vec3 ground = mix(grass, snowG, clamp(uWinter, 0.0, 1.0));
    ground = mix(ground, ground * 0.6, uRain * 0.6);  // wet
    vec3 c = ground * (mix(L.sky1 * 0.6, L.sun, 0.5) * L.amb + L.sky0 * 0.1);
    // Winding river: centre line meanders with distance; ripples flow toward the viewer.
    float cx = uRiverX * aspect * z * 3.0 + sin(z * 0.9 + 1.3) * 0.9 + sin(z * 0.37) * 0.6;
    float w = 0.55;
    float dist = abs(g.x - cx);
    float mask = smoothstep(w, w * 0.8, dist) * clamp(uRiver, 0.0, 1.0);
    if (mask > 0.001) {
      vec2 rip = vec2(fbm(vec2(g.x * 3.0, z * 6.0 - uTime * 0.6)) - 0.5, 0.0) * 0.012;
      vec2 refl = vec2(uv.x, 2.0 * floorY - uv.y) + rip;
      vec3 r = backdrop(clamp(refl, 0.0, 1.0), L, aspect);
      vec3 water = mix(vec3(0.03, 0.07, 0.09), r * 0.8, 0.7);
      water = mix(water, vec3(0.66, 0.72, 0.78) * L.amb + r * 0.2, clamp(uWinter - 0.7, 0.0, 0.6));  // ice
      float glint = pow(max(0.0, fbm(vec2(g.x * 8.0, z * 20.0 - uTime * 1.2)) - 0.55) * 2.5, 3.0);
      water += L.sun * glint * 0.45;
      float bank = smoothstep(w * 0.6, w, dist);  // dark wet edge near the banks
      c = mix(c, mix(water, c * 0.55, bank * 0.5), mask);
    }
    float air = clamp(1.0 - 1.0 / (1.0 + z * 0.08), 0.0, 1.0) * 0.6 + uFog * 0.5;
    col = mix(c, L.haze * L.amb, clamp(air, 0.0, 0.9));
  }
  // Foreground pine clusters at the frame edges: the nearest depth plane.
  {
    vec2 u = parallax(uv, 1.35);
    float x = (u.x - 0.5) * aspect;
    float edge = smoothstep(0.2, 0.5, abs(u.x - 0.5));
    float gy = 0.03 + 0.05 * vnoise1(x * 2.0 + 4.0);
    if (edge > 0.01) {
      float t = u.y - gy;
      if (t < 0.0) {
        col = mix(col, vec3(0.03, 0.04, 0.035) * L.amb + clamp(uWinter, 0.0, 1.0) * 0.35 * L.amb, smoothstep(0.3, 0.7, edge));
      } else if (pines(x, t, 6.0, 51.0, 0.65 * edge, 0.0) > 0.5) {
        col = pineColor(t, x, L) * 0.8;
      }
    }
  }
  // Falling snow, three parallax layers; intensity above 1 thickens it.
  float snowAmt = clamp(uWinter, 0.0, 1.3);
  if (snowAmt > 0.01) {
    for (int k = 0; k < 3; k++) {
      float fk = float(k);
      float scale = mix(9.0, 26.0, fk / 2.0);
      vec2 sp = vec2(uv.x * aspect, uv.y) * scale;
      sp.y += uTime * mix(1.6, 0.9, fk / 2.0);
      sp.x += sin(uTime * 0.7 + sp.y * 0.35 + fk) * 0.35 + uTilt.x * (2.0 - fk);
      vec2 id = floor(sp);
      float rnd = hash12(id + fk * 13.0);
      vec2 c0 = vec2(hash12(id + 3.1), hash12(id + 7.7));
      float d = length(fract(sp) - c0);
      float flake = smoothstep(mix(0.11, 0.05, fk / 2.0), 0.0, d) * step(1.0 - 0.55 * snowAmt, rnd);
      col = mix(col, vec3(0.93, 0.94, 0.96) * mix(1.0, 0.55, uNight), flake * mix(0.85, 0.45, fk / 2.0));
    }
  }
  // Rain streaks.
  if (uRain > 0.01) {
    vec2 rp = vec2(uv.x * aspect + uv.y * 0.18, uv.y) * vec2(70.0, 2.2);
    rp.y += uTime * 5.5;
    vec2 id = floor(rp);
    float streak = step(1.0 - 0.35 * uRain, hash12(id)) * smoothstep(0.08, 0.0, abs(fract(rp.x) - 0.5)) * smoothstep(0.0, 0.6, fract(rp.y));
    col = mix(col, vec3(0.72, 0.76, 0.80) * L.amb, streak * 0.3);
  }
  // Global fog veil.
  col = mix(col, L.haze * L.amb, uFog * 0.35 * smoothstep(0.0, 0.6, 1.0 - abs(uv.y - HORIZON) * 1.6));
  // Filmic grade: partial desaturation, lifted blacks, cool shadows and warm highlights.
  float luma = dot(col, vec3(0.299, 0.587, 0.114));
  col = mix(vec3(luma), col, 0.78);
  col = mix(col * vec3(0.92, 0.98, 1.04), col * vec3(1.04, 1.0, 0.94), smoothstep(0.2, 0.8, luma));
  col = col * 0.95 + vec3(0.018, 0.022, 0.026);
  gl_FragColor = vec4(col, 1.0);
}
`;
