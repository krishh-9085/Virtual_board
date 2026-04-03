import React, { useEffect, useRef, useState } from 'react';
// Using window variables for MediaPipe due to Vite compatibility
import { analyzeGestures } from './utils/gestureLogic';
import { PointerSmoother } from './utils/smoothing';
import { CanvasManager } from './utils/CanvasManager';
// Removed framer-motion due to eslint constraints
import { 
  Eraser, Undo, Redo, 
  Trash2, Download, Settings, Loader2, MousePointer2, Wand
} from 'lucide-react';
import './VirtualBoard.css';

const DEFAULT_COLORS = ['#FFFFFF', '#FF3B30', '#34C759', '#007AFF', '#FFCC00', '#FF9500'];

export default function VirtualBoard() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const pointerCanvasRef = useRef(null);
  
  const [manager, setManager] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [mode, setMode] = useState('idle');
  const [fps, setFps] = useState(0);
  
  // UI State
  const [brushColor, setBrushColor] = useState(DEFAULT_COLORS[0]);
  const [brushSize, setBrushSize] = useState(5);
  const [isEraserMode, setIsEraserMode] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [magicShape, setMagicShape] = useState(true); // Default to ON

  // Tracking state refs (to avoid stale closures in mediapipe callback)
  const stateRef = useRef({
    brushColor,
    brushSize,
    isEraserMode,
    isDrawing: false,
    activeGesture: 'idle',
    lastFrameTime: performance.now(),
    hoveredBtn: null,
  });

  // Hot sync magicShape
  useEffect(() => {
    if (manager) manager.magicShape = magicShape;
  }, [magicShape, manager]);

  // Hot sync magicShape
  useEffect(() => {
    if (manager) manager.magicShape = magicShape;
  }, [magicShape, manager]);

  // Sync state to ref
  useEffect(() => {
    stateRef.current = { brushColor, brushSize, isEraserMode, isDrawing: stateRef.current.isDrawing, lastFrameTime: stateRef.current.lastFrameTime, hoveredBtn: stateRef.current.hoveredBtn };
  }, [brushColor, brushSize, isEraserMode]);

  useEffect(() => {
    if (!canvasRef.current || !pointerCanvasRef.current || !videoRef.current) return;

    // Set canvas dimensions
    const resizeCanvas = () => {
      const parent = canvasRef.current.parentElement;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      canvasRef.current.width = w;
      canvasRef.current.height = h;
      pointerCanvasRef.current.width = w;
      pointerCanvasRef.current.height = h;
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const canvasManager = new CanvasManager(canvasRef.current);
    setManager(canvasManager);
    
    // Alpha 0.55 is the mathematical sweet spot between real-time tracking and silky smooth curved handwriting.
    const smoother = new PointerSmoother(0.55); 
    
    const Hands = window.Hands;
    const Camera = window.Camera;
    
    // Initialize MediaPipe Hands
    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      // Lowered from strict 0.85 to 0.60 so it doesn't lose tracking if lighting drops or hand tilts
      minDetectionConfidence: 0.60,
      minTrackingConfidence: 0.60
    });

    let frameCount = 0;
    
    hands.onResults((results) => {
      // Calculate FPS
      const now = performance.now();
      const dt = now - stateRef.current.lastFrameTime;
      if (dt > 1000) {
        setFps(Math.round((frameCount * 1000) / dt));
        frameCount = 0;
        stateRef.current.lastFrameTime = now;
      }
      frameCount++;

      const pCtx = pointerCanvasRef.current.getContext('2d');
      // Clear pointer canvas
      pCtx.clearRect(0, 0, pointerCanvasRef.current.width, pointerCanvasRef.current.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        setIsLoading(false);
        const landmarks = results.multiHandLandmarks[0];
        
        // Use Index Finger Tip for drawing
        const indexTip = landmarks[8];
        const rawX = (1 - indexTip.x) * pointerCanvasRef.current.width; // Mirror X
        const rawY = indexTip.y * pointerCanvasRef.current.height;
        
        const currentGesture = analyzeGestures(landmarks, stateRef.current.isDrawing);
        let displayMode = currentGesture;
        
        // Check UI overrides
        const actualEraserMode = currentGesture === 'erase' || stateRef.current.isEraserMode;
        if (stateRef.current.isEraserMode && currentGesture === 'draw') {
            displayMode = 'erase-draw';
        }

        if (currentGesture === 'pinch') {
          const element = document.elementFromPoint(rawX, rawY);
          if (element) {
            if (element.classList.contains('color-btn')) {
              const color = element.getAttribute('data-color');
              if (color && stateRef.current.brushColor !== color) {
                setBrushColor(color);
                setIsEraserMode(false);
              }
            } else if (element.classList.contains('size-slider')) {
              const rect = element.getBoundingClientRect();
              const relativeX = rawX - rect.left;
              const percentage = Math.max(0, Math.min(1, relativeX / rect.width));
              const min = parseInt(element.min) || 2;
              const max = parseInt(element.max) || 30;
              const newSize = Math.round(min + percentage * (max - min));
              if (stateRef.current.brushSize !== newSize) {
                  setBrushSize(newSize);
              }
            } else {
              const btn = element.closest('.icon-btn');
              if (btn) {
                if (stateRef.current.hoveredBtn !== btn) {
                    stateRef.current.hoveredBtn = btn;
                    btn.click();
                    setTimeout(() => {
                        if (stateRef.current.hoveredBtn === btn) {
                            stateRef.current.hoveredBtn = null;
                        }
                    }, 500); // 500ms debounce
                }
              }
            }
          }
        } else {
            stateRef.current.hoveredBtn = null;
        }

        // --- GESTURE STATE CUTOFF ---
        // If we switch directly between draw <-> erase without going idle, we must forcibly end the stroke!
        if (stateRef.current.isDrawing && stateRef.current.activeGesture !== currentGesture) {
            canvasManager.endStroke();
            stateRef.current.isDrawing = false;
            smoother.reset();
        }
        stateRef.current.activeGesture = currentGesture;

        // Render Pointer cursor
        const pCtx = pointerCanvasRef.current.getContext('2d');
        pCtx.clearRect(0, 0, pointerCanvasRef.current.width, pointerCanvasRef.current.height);

        // Draw normal dot cursor for drawing / hovering
        if (currentGesture === 'draw' || currentGesture === 'pinch') {
            pCtx.beginPath();
            pCtx.arc(rawX, rawY, currentGesture === 'pinch' ? 8 : 4, 0, 2 * Math.PI);
            pCtx.fillStyle = stateRef.current.isEraserMode ? 'rgba(255,255,255,0.5)' : stateRef.current.brushColor;
            pCtx.fill();
            if (currentGesture === 'pinch') {
              pCtx.strokeStyle = 'white';
              pCtx.lineWidth = 2;
              pCtx.stroke();
            }
        } 
        // Draw square bounding brush for Palm Eraser
        else if (currentGesture === 'erase') {
            const palmNode = landmarks[9];
            const rawPalmX = (1 - palmNode.x) * pointerCanvasRef.current.width;
            const rawPalmY = palmNode.y * pointerCanvasRef.current.height;
            const eSize = 60; // Big eraser box
            pCtx.beginPath();
            pCtx.rect(rawPalmX - eSize/2, rawPalmY - eSize/2, eSize, eSize);
            pCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            pCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            pCtx.lineWidth = 2;
            pCtx.fill();
            pCtx.stroke();
        }

        // Logic Actions
        if (currentGesture === 'draw') {
          canvasManager.magicShape = stateRef.current.magicShape;
          const { x, y } = smoother.smooth(rawX, rawY);
          
          if (!stateRef.current.isDrawing) {
            stateRef.current.isDrawing = true;
            canvasManager.startStroke(x, y, stateRef.current.brushColor, stateRef.current.brushSize, stateRef.current.isEraserMode);
          } else {
            canvasManager.continueStroke(x, y);
          }
        } 
        else if (currentGesture === 'erase') {
          // Trigger actual erasing tied to the Palm position
          const palmNode = landmarks[9];
          const rawPalmX = (1 - palmNode.x) * pointerCanvasRef.current.width;
          const rawPalmY = palmNode.y * pointerCanvasRef.current.height;
          // Notice we don't use 'smoother' here, just direct raw because palm doesn't jitter like fingertips
          
          if (!stateRef.current.isDrawing) {
            stateRef.current.isDrawing = true;
            canvasManager.startStroke(rawPalmX, rawPalmY, '#0', 60, true);
          } else {
            canvasManager.continueStroke(rawPalmX, rawPalmY);
          }
        }
        else if (currentGesture === 'pause' || currentGesture === 'idle' || currentGesture === 'pinch') {
            canvasManager.endStroke();
            stateRef.current.isDrawing = false;
        }

        setMode(actualEraserMode ? 'erase' : displayMode);
      } else {
        smoother.reset();
        if (stateRef.current.isDrawing) {
          canvasManager.endStroke();
          stateRef.current.isDrawing = false;
        }
        setMode('idle');
      }
    });

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 1280,
      height: 720
    });

    camera.start();

    return () => {
      camera.stop();
      hands.close();
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  const handleExport = () => {
    if (!manager) return;
    const url = manager.exportDataURL();
    const a = document.createElement('a');
    a.href = url;
    a.download = `VirtualBoard_${new Date().getTime()}.png`;
    a.click();
  };

  return (
    <div className="board-container dark-theme">
      {isLoading && (
        <div className="loading-overlay">
          <Loader2 className="spinner" size={48} />
          <p>Initializing Camera & AI Tracking...</p>
        </div>
      )}

      {/* Main Work Area */}
      <div className="workspace">
        <video 
          ref={videoRef} 
          className="camera-feed"
          autoPlay 
          playsInline
        />
        <canvas ref={canvasRef} className="drawing-layer" />
        <canvas ref={pointerCanvasRef} className={`pointer-layer ${showGrid ? 'show-grid' : ''}`} />
      </div>

      {/* Floating Header info */}
      <header className="board-header">
        <div className="status-badge">
          <span className={`indicator ${isLoading ? 'yellow' : 'green'}`} />
          <span>{isLoading ? 'Booting' : 'Live'}</span>
        </div>
        <div className="status-badge pulse">
          <MousePointer2 size={16}/>
          <span style={{textTransform: 'uppercase'}}>{mode}</span>
        </div>
        {fps > 0 && (
          <div className="fps-counter">
            {fps} FPS
          </div>
        )}
      </header>

      {/* Floating UI Toolbar */}
      <div 
        className="floating-toolbar"
      >
        <div className="tool-section">
          {DEFAULT_COLORS.map(c => (
             <button 
               key={c}
               data-color={c}
               className={`color-btn ${brushColor === c && !isEraserMode ? 'active' : ''}`}
               style={{ backgroundColor: c }}
               onClick={() => { setBrushColor(c); setIsEraserMode(false); }}
             />
          ))}
        </div>
        
        <div className="divider" />
        
        <div className="tool-section">
          <input 
            type="range" 
            min="2" max="30" 
            value={brushSize} 
            onChange={e => setBrushSize(parseInt(e.target.value))}
            className="size-slider"
          />
        </div>

        <div className="divider" />

        <div className="tool-section">
          <button 
             className={`icon-btn ${isEraserMode ? 'active' : ''}`} 
             onClick={() => setIsEraserMode(!isEraserMode)}
             title="Eraser Toggle"
          >
             <Eraser size={20} />
          </button>
          <button className="icon-btn" onClick={() => manager?.undo()} title="Undo">
             <Undo size={20} />
          </button>
          <button className="icon-btn" onClick={() => manager?.redo()} title="Redo">
             <Redo size={20} />
          </button>
          <button className="icon-btn danger" onClick={() => manager?.clear()} title="Clear Board">
             <Trash2 size={20} />
          </button>
        </div>

        <div className="divider" />

        <div className="tool-section">
          <button className={`icon-btn ${magicShape ? 'active' : ''}`} onClick={() => setMagicShape(!magicShape)} title="Magic Shape Gen">
             <Wand size={20} />
          </button>
          <button className={`icon-btn ${showGrid ? 'active' : ''}`} onClick={() => setShowGrid(!showGrid)} title="Toggle Grid">
             <Settings size={20} />
          </button>
          <button className="icon-btn" onClick={handleExport} title="Download PNG">
             <Download size={20} />
          </button>
        </div>

      </div>
    </div>
  );
}
