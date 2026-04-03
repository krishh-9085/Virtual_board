export function getDistance(p1, p2) {
  // Use 3D distance because users angle their hands while drawing!
  const z1 = p1.z || 0;
  const z2 = p2.z || 0;
  return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (z1 - z2) ** 2);
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

  // Finger PIPs (middle joints)
  const indexPip = landmarks[6];
  const middlePip = landmarks[10];
  const ringPip = landmarks[14];
  const pinkyPip = landmarks[18];

  // Scale reference
  const middleMcp = landmarks[9];
  const handSize = getDistance(wrist, middleMcp);
  
  // Pinch detection (to prevent drawing when pinched)
  const pinchDistance = getDistance(thumbTip, indexTip);
  const isPinching = pinchDistance < handSize * 0.5;

  // 3D Joint Check: if a finger is extended, the tip physically cannot be closer to the wrist than its middle joint.
  // This is mathematically immune to tilt, angle, or camera perspective!
  const isIndexUp = getDistance(wrist, indexTip) > getDistance(wrist, indexPip);
  const isMiddleUp = getDistance(wrist, middleTip) > getDistance(wrist, middlePip);
  const isRingUp = getDistance(wrist, ringTip) > getDistance(wrist, ringPip);
  const isPinkyUp = getDistance(wrist, pinkyTip) > getDistance(wrist, pinkyPip);

  if (isPinching) {
    return 'pinch'; // Return pinch for UI interaction
  }

  // Count how many of the 4 fingers are up
  const openFingersCount = [isIndexUp, isMiddleUp, isRingUp, isPinkyUp].filter(Boolean).length;

  // Erase: At least 3 of the 4 main fingers extended
  if (openFingersCount >= 3) {
    return 'erase';
  }

  // Draw: ONLY Index finger extended
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    return 'draw';
  }

  return 'idle';
}
