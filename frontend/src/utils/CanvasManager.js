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
    
    this.ctx.beginPath();
    const pts = this.currentStroke.points;
    this.ctx.moveTo(pts[pts.length - 2].x, pts[pts.length - 2].y);
    this.applyStrokeStyles(this.currentStroke);
    this.ctx.lineTo(x, y);
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
    
    // 2. Closed Shape Test -> Tight 25% diagonal bounding threshold (gap must be small)
    const startEndDist = Math.sqrt(Math.pow(pStart.x - pEnd.x, 2) + Math.pow(pStart.y - pEnd.y, 2));
    const boundingDiag = Math.sqrt(width*width + height*height);
    
    if (startEndDist < boundingDiag * 0.25) {
        // Shoelace Polygon Area
        let area = 0;
        for(let i=0; i<pts.length-1; i++) {
            area += pts[i].x * pts[i+1].y - pts[i+1].x * pts[i].y;
        }
        area += pts[pts.length-1].x * pStart.y - pStart.x * pts[pts.length-1].y;
        area = Math.abs(area / 2);
        
        // Circularity using the jitter-free sampledPerim
        const circularity = (4 * Math.PI * area) / (sampledPerim * sampledPerim);
        
        // Circle (Strict > 0.85)
        if (circularity > 0.85) {
            const avgD = (width + height) / 2;
            const cx = minX + width / 2;
            const cy = minY + height / 2;
            const radius = avgD / 2;
            stroke.points = [];
            for(let i=0; i<=36; i++) {
                const angle = i * (Math.PI * 2 / 36);
                stroke.points.push({ x: cx + Math.cos(angle)*radius, y: cy + Math.sin(angle)*radius });
            }
            stroke.isGeometry = true;
            return;
        }
        
        // Rectangle/Square (Strict between 0.70 and 0.85) (Perfect square is 0.78)
        if (circularity > 0.70 && circularity <= 0.85) {
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
        
        // Triangle (Strict between 0.50 and 0.65) (Perfect equilateral is 0.60)
        if (circularity > 0.50 && circularity <= 0.65) {
            // Snap to isosceles triangle pointing up/down based on bounding box
            stroke.points = [
                {x: minX + width/2, y: minY},
                {x: maxX, y: maxY},
                {x: minX, y: maxY},
                {x: minX + width/2, y: minY}
            ];
            stroke.isGeometry = true;
            return;
        }
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
      if (pts.length > 0) {
          this.ctx.moveTo(pts[0].x, pts[0].y);
          for(let i=1; i<pts.length; i++) {
              this.ctx.lineTo(pts[i].x, pts[i].y);
          }
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
