export class CanvasManager {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.history = [];
    this.redoStack = [];
    this.currentStroke = null;
    this.magicShape = true; // Default to TRUE
    
    // Set default styles
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    
    // Auto-load history
    const saved = localStorage.getItem('virtual_board_strokes');
    if (saved) {
        try {
            this.history = JSON.parse(saved);
            // Delay redraw slightly to ensure canvas DOM size is set
            setTimeout(() => this.redrawAll(), 50);
        } catch(e) { }
    }
  }

  // Auto-save util throttled
  saveState() {
     clearTimeout(this.saveTimer);
     this.saveTimer = setTimeout(() => {
         localStorage.setItem('virtual_board_strokes', JSON.stringify(this.history));
     }, 1000); // 1-second debounce
  }

  startStroke(x, y, color, size, isEraser = false) {
    this.currentStroke = {
      color: color,
      size: size,
      isEraser: isEraser,
      points: [{ x, y }]
    };
    
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
    this.applyStrokeStyles(this.currentStroke);
    
    // Draw a single dot if it's just a tap
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
  }

  continueStroke(x, y) {
    if (!this.currentStroke) return;
    
    this.currentStroke.points.push({ x, y });
    const pts = this.currentStroke.points;
    
    this.ctx.beginPath();
    
    // Smooth spline interpolation for gorgeous ink flow
    if (pts.length > 2) {
        const p1 = pts[pts.length - 3];
        const p2 = pts[pts.length - 2];
        const p3 = pts[pts.length - 1];
        
        const mid1 = { x: (p1.x + p2.x)/2, y: (p1.y + p2.y)/2 };
        const mid2 = { x: (p2.x + p3.x)/2, y: (p2.y + p3.y)/2 };
        
        this.ctx.moveTo(mid1.x, mid1.y);
        this.applyStrokeStyles(this.currentStroke);
        this.ctx.quadraticCurveTo(p2.x, p2.y, mid2.x, mid2.y);
    } else if (pts.length === 2) {
        this.ctx.moveTo(pts[0].x, pts[0].y);
        this.applyStrokeStyles(this.currentStroke);
        this.ctx.lineTo(pts[1].x, pts[1].y);
    }
    this.ctx.stroke();
  }

  endStroke() {
    if (!this.currentStroke) return;

    if (this.magicShape && !this.currentStroke.isEraser) {
        this.recognizeShapeAndSnap(this.currentStroke);
    }
    
    this.history.push(this.currentStroke);
    this.currentStroke = null;
    this.redoStack = []; // Clear redo stack on new action
    this.saveState();
    this.redrawAll();
  }

  recognizeShapeAndSnap(stroke) {
    const pts = stroke.points;
    if (pts.length < 10) return;

    const minX = Math.min(...pts.map(p => p.x));
    const maxX = Math.max(...pts.map(p => p.x));
    const minY = Math.min(...pts.map(p => p.y));
    const maxY = Math.max(...pts.map(p => p.y));
    
    const width = maxX - minX;
    const height = maxY - minY;
    
    const pStart = pts[0];
    const pEnd = pts[pts.length - 1];
    const directDist = Math.sqrt(Math.pow(pEnd.x - pStart.x, 2) + Math.pow(pEnd.y - pStart.y, 2));

    // Calculate a Jitter-Free Perimeter by sampling ~20 points
    let sampledPerim = 0;
    const step = Math.max(1, Math.floor(pts.length / 20));
    let lastP = pts[0];
    for (let i = step; i < pts.length; i += step) {
        sampledPerim += Math.sqrt((pts[i].x - lastP.x)**2 + (pts[i].y - lastP.y)**2);
        lastP = pts[i];
    }
    sampledPerim += Math.sqrt((pEnd.x - lastP.x)**2 + (pEnd.y - lastP.y)**2);

    // 1. Line Test (Strict 85% line similarity)
    if (sampledPerim < directDist * 1.15) {
        stroke.points = [pStart, pEnd];
        stroke.isGeometry = true;
        return;
    }
    
    // 2. Closed Shape Test -> 35% bounding threshold
    const startEndDist = Math.sqrt(Math.pow(pStart.x - pEnd.x, 2) + Math.pow(pStart.y - pEnd.y, 2));
    const boundingDiag = Math.sqrt(width*width + height*height);
    
    if (startEndDist < boundingDiag * 0.40) { // Gap tolerance 40%
        // Evaluate Radius Variance (Circle vs Polygon)
        const cx = minX + width / 2;
        const cy = minY + height / 2;
        const avgR = (width + height) / 4;
        
        let variance = 0;
        for(let p of pts) {
            const distToCenter = Math.sqrt((p.x - cx)**2 + (p.y - cy)**2);
            variance += Math.abs(distToCenter - avgR);
        }
        variance /= pts.length;
        
        // --- 1. Circle Test ---
        // If variance is small, it stayed roughly equidistant from the center
        if (variance < avgR * 0.30) {
            const radius = avgR;
            stroke.points = [];
            for(let i=0; i<=36; i++) {
                const angle = i * (Math.PI * 2 / 36);
                stroke.points.push({ x: cx + Math.cos(angle)*radius, y: cy + Math.sin(angle)*radius });
            }
            stroke.isGeometry = true;
            return;
        }
        
        // --- 2. Rectangle/Square Snap ---
        // If it was closed but NOT a circle, snap it to its bounding box
        stroke.points = [
            {x: minX, y: minY},
            {x: maxX, y: minY},
            {x: maxX, y: maxY},
            {x: minX, y: maxY},
            {x: minX, y: minY}
        ];
        stroke.isGeometry = true;
        return;
    }
  }

  applyStrokeStyles(stroke) {
    this.ctx.lineWidth = stroke.size;
    if (stroke.isEraser) {
      this.ctx.globalCompositeOperation = 'destination-out';
      this.ctx.strokeStyle = 'rgba(0,0,0,1)';
    } else {
      this.ctx.globalCompositeOperation = 'source-over';
      this.ctx.strokeStyle = stroke.color;
    }
  }

  redrawAll() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const stroke of this.history) {
      if (stroke.points.length === 0) continue;
      
      this.applyStrokeStyles(stroke);
      this.ctx.beginPath();
      
      const pts = stroke.points;
      if (pts.length > 2) {
          this.ctx.moveTo(pts[0].x, pts[0].y);
          for(let i=1; i<pts.length-1; i++) {
              const mid = { x: (pts[i].x + pts[i+1].x)/2, y: (pts[i].y + pts[i+1].y)/2 };
              this.ctx.quadraticCurveTo(pts[i].x, pts[i].y, mid.x, mid.y);
          }
          this.ctx.lineTo(pts[pts.length-1].x, pts[pts.length-1].y);
      } else if (pts.length === 2) {
          this.ctx.moveTo(pts[0].x, pts[0].y);
          this.ctx.lineTo(pts[1].x, pts[1].y);
      }
      this.ctx.stroke();
    }
  }

  undo() {
    if (this.history.length === 0) return;
    const lastStroke = this.history.pop();
    this.redoStack.push(lastStroke);
    this.redrawAll();
    this.saveState();
  }

  redo() {
    if (this.redoStack.length === 0) return;
    const stroke = this.redoStack.pop();
    this.history.push(stroke);
    this.redrawAll();
    this.saveState();
  }

  clear() {
    this.history = [];
    this.redoStack = [];
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.saveState();
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.redrawAll(); // Need to redraw after resizing clears canvas natively
  }

  exportDataURL() {
    return this.canvas.toDataURL('image/png');
  }
}
