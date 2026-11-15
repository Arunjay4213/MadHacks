import { Component, type ReactNode } from 'react';

export default class OpticalBoundary extends Component<{ children: ReactNode; onError?: () => void; fallback?: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch() { this.props.onError?.(); }
  render() {
    if (!this.state.failed) return this.props.children;
    return this.props.fallback !== undefined ? this.props.fallback : <div className="optical-fallback" role="status">
      <p>The light needs a little more room.</p><span>WebGL is unavailable. Try a hardware-accelerated browser; keyboard commands still work.</span>
    </div>;
  }
}
