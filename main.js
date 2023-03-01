const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture-button');
const capturedImages = document.getElementById('captured-images');
const landmarkModel = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    { shouldLoadIrisModel: false }
);
const expressionModel = await faceExpressionRecognition.load(
    faceExpressionRecognition.SupportedPackages.TF_2
);

let stream;
let faceMesh;

async function setupCamera() {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
}

async function captureFace() {
    // Draw the current frame from the video element onto the canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Detect face landmarks using the mediapipeFacemesh model
    const faceLandmarks = await landmarkModel.estimateFaces({ input: video });
    if (faceLandmarks.length === 0) {
        alert('No face detected!');
        return;
    }

    // Crop the face from the video frame using the face landmarks
    const boundingBox = getFaceBoundingBox(faceLandmarks[0].landmarks);
    const croppedImage = getCroppedImage(boundingBox);

    // Recognize facial expression using the TF_2 model
    const expression = await expressionModel.predict(croppedImage);
    console.log(expression);

    // Display the captured face in the UI
    const image = new Image();
    image.src = croppedImage.toDataURL();
    image.classList.add('captured-image');
    capturedImages.appendChild(image);
}

function getFaceBoundingBox(faceLandmarks) {
    // Get the x and y coordinates of all the face landmarks
    const xCoords = faceLandmarks.map(landmark => landmark[0]);
    const yCoords = faceLandmarks.map(landmark => landmark[1]);

    // Find the minimum and maximum x and y coordinates
    const xMin = Math.min(...xCoords);
    const xMax = Math.max(...xCoords);
    const yMin = Math.min(...yCoords);
    const yMax = Math.max(...yCoords);

    // Calculate the width and height of the bounding box
    const width = xMax - xMin;
    const height = yMax - yMin;

    // Return the bounding box as an object with x, y, width, and height properties
    return {
        x: xMin,
        y: yMin,
        width: width,
        height: height
    };
}

