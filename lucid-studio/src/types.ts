export type ModifierKind = 'winter' | 'river' | 'night' | 'sunset' | 'rain' | 'fog';
export interface SpatialCoords { x: number; y: number }
export interface ActiveModifier {
  id: string;
  label: string;
  intensity: number;
  coords?: SpatialCoords;
}
export interface MutationPayload {
  prompt: string;
  activeState: string[];
  spatialCoords?: SpatialCoords | null;
  timestamp: number;
}
export interface WorldStreamAdapter {
  sendMutation: (payload: MutationPayload) => Promise<void>;
}
export interface MutationImpulse {
  id: number;
  startedAt: number; // performance.now(), shared by optical and particle effects
  localized: boolean;
  center?: SpatialCoords;
}
export interface AudioBands { low: number; vocal: number; high: number }
export interface WorldCanvasProps {
  modifiers: ActiveModifier[];
  source?: MediaStream | HTMLCanvasElement | null;
  impulse: MutationImpulse | null;
  pointer: SpatialCoords; // normalized 0..1
  reducedMotion?: boolean;
  active?: boolean;
  onReady?: (preview?: string) => void;
}
export interface AcousticSingularityProps {
  bands: AudioBands;
  impulse: MutationImpulse | null;
  reducedMotion?: boolean;
}
