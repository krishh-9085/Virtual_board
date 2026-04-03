export class PointerSmoother {
  constructor(alpha = 0.35) {
    this.alpha = alpha; // Weight for new points (lower = smoother but more lag)
    this.smoothedX = null;
    this.smoothedY = null;
  }

  smooth(x, y) {
    if (this.smoothedX === null || this.smoothedY === null) {
      this.smoothedX = x;
      this.smoothedY = y;
    } else {
      this.smoothedX = this.alpha * x + (1 - this.alpha) * this.smoothedX;
      this.smoothedY = this.alpha * y + (1 - this.alpha) * this.smoothedY;
    }
    return { x: this.smoothedX, y: this.smoothedY };
  }

  reset() {
    this.smoothedX = null;
    this.smoothedY = null;
  }
  
  setAlpha(newAlpha) {
      this.alpha = newAlpha;
  }
}
