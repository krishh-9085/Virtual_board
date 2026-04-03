export function getDistance(p1, p2) {
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

export function analyzeGestures(landmarks) {
  if (!landmarks || landmarks.length < 21) return 'idle';

  const wrist = landmarks[0];
  
  // Finger Tips
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  const indexMcp = landmarks[5];
  const ringMcp = landmarks[13];
  const pinkyMcp = landmarks[17];

  // Scale reference
  const middleMcp = landmarks[9];
  const handSize = getDistance(wrist, middleMcp);
  
  // Pinch detection (to prevent drawing when pinched)
  const pinchDistance = getDistance(thumbTip, indexTip);
  const isPinching = pinchDistance < handSize * 0.5;

  // Check if fingers are extended (Tip distance to Base Knuckle)
  // When a finger is curled into a fist, the tip distance to MCP drops to almost 0.
  // When extended, it stretches to the full length of the finger (> 0.8 * handSize)
  const isIndexUp = getDistance(indexTip, indexMcp) > handSize * 0.8;
  const isMiddleUp = getDistance(middleTip, middleMcp) > handSize * 0.8;
  const isRingUp = getDistance(ringTip, ringMcp) > handSize * 0.8;
  const isPinkyUp = getDistance(pinkyTip, pinkyMcp) > handSize * 0.8;

  if (isPinching) {
    return 'pinch'; // Return pinch for UI interaction
  }

  // Count how many of the 4 fingers are up
  const openFingersCount = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

  // Erase: At least 3 of the 4 main fingers extended (more robust palm detection)
  if (openFingersCount >= 3) {
    return 'erase';
  }

  // Draw: ONLY Index finger extended
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    return 'draw';
  }

  return 'idle';
}
