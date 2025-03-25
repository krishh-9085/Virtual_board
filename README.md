# Virtual Writing Board using Hand Gestures

This project enables a **Virtual Writing Board** where users can draw and erase on the screen using simple **hand gestures** captured by the webcam. The application utilizes the **MediaPipe** library to detect hand movements and **OpenCV** for image processing.

## Features
- **Color Selection**: Choose from a set of colors (White, Black, Blue, Green, Red) by pinching your thumb and index finger.
- **Drawing Mode**: Draw by raising your index finger and moving it across the screen.
- **Erase Mode**: Erase your drawing by fully extending your hand.
- **Clear Board**: Reset the canvas with the "Clear" button.
- **Camera Support**: Automatically detects and allows you to choose from available cameras on your system.

## Libraries Used
- **OpenCV**: For video capture, image processing, and displaying output.
- **MediaPipe**: For real-time hand tracking and gesture recognition.
- **NumPy**: For canvas management and performing mathematical operations.

## Usage

1. **Run the script** to activate your webcam and start the drawing interface.
2. **Select a camera**: If multiple cameras are available, choose one from the available list.
3. **Use hand gestures**:
   - **Pinch Gesture**: Pinch your thumb and index finger to select a color (White, Black, Blue, Green, or Red).
   - **Drawing**: Raise your index finger to draw.
   - **Erasing**: Open your hand to erase your drawing.

## Controls
- **'q'**: Exit the application.
- **'c'**: Clear the drawing board.

## Color Palette
- **White**
- **Black**
- **Blue**
- **Green**
- **Red**
- **Clear (Reset Canvas)**

## Requirements

- Python 3.9
- OpenCV
- MediaPipe
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/krishh-9085/virtual-writing-board.git
