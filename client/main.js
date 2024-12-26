// Get elements
const startCameraButton = document.getElementById("startCamera");
const takePhotoButton = document.getElementById("takePhoto");
const submitButton = document.getElementById("submit");
const video = document.getElementById("video");
const registerForm = document.getElementById("registerForm");
const cameraButtons = [startCameraButton, takePhotoButton, submitButton];

// Camera initialization
let stream;

startCameraButton.addEventListener("click", startCamera);
takePhotoButton.addEventListener("click", takePhoto);
submitButton.addEventListener("click", submitForm);

// Start camera
function startCamera() {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((mediaStream) => {
      stream = mediaStream;
      video.srcObject = mediaStream;
      startCameraButton.style.display = "none";
      takePhotoButton.style.display = "inline-block";
      submitButton.style.display = "inline-block";
    })
    .catch((err) => console.error("Camera access denied:", err));
}

// Take photo
function takePhoto() {
  // Here you would add the code to take the photo
  // For now, we'll simulate it
  console.log("Photo taken");
}

// Submit the form
function submitForm(event) {
  event.preventDefault();
  // Submit form logic here (like saving data to database)
  alert("Form Submitted");
  stopCamera();
}

// Stop camera
function stopCamera() {
  if (stream) {
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
  }
  video.srcObject = null;
  startCameraButton.style.display = "inline-block";
  takePhotoButton.style.display = "none";
  submitButton.style.display = "none";
}
