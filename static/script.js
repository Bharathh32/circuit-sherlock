
// ------------ ELEMENTS ------------
const imageDrop = document.getElementById("imageDrop");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");

// ------------ CLICK EVENTS ------------
imageDrop.onclick = () => imageInput.click();

// ------------ DRAG & DROP ------------
imageDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
    imageDrop.classList.add("active");
});

imageDrop.addEventListener("dragleave", () => {
    imageDrop.classList.remove("active");
});

imageDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    imageDrop.classList.remove("active");

    const file = e.dataTransfer.files[0];
    handleImage(file);
});

// ------------ WHEN USER SELECTS FILE ------------
imageInput.addEventListener("change", (e) => {
    handleImage(e.target.files[0]);
});

// ------------ SHOW PREVIEW ------------
function handleImage(file) {
    if (!file) return;

    const reader = new FileReader();

    reader.onload = () => {
        imagePreview.src = reader.result;
        imagePreviewContainer.style.display = "block";
    };

    reader.readAsDataURL(file);
}



// // VIDEO ELEMENTS
// const videoDrop = document.getElementById("videoDrop");
// const videoInput = document.getElementById("videoInput");
// const videoPreview = document.getElementById("videoPreview");
// const videoPreviewContainer = document.getElementById("videoPreviewContainer");

// // CLICK TO SELECT VIDEO
// videoDrop.onclick = () => videoInput.click();

// // DRAG & DROP VIDEO
// videoDrop.addEventListener("dragover", (e) => {
//     e.preventDefault();
//     videoDrop.classList.add("active");
// });

// videoDrop.addEventListener("dragleave", () => {
//     videoDrop.classList.remove("active");
// });

// videoDrop.addEventListener("drop", (e) => {
//     e.preventDefault();
//     videoDrop.classList.remove("active");

//     const file = e.dataTransfer.files[0];
//     handleVideo(file);
// });

// // VIDEO INPUT CHANGE
// videoInput.addEventListener("change", (e) => {
//     handleVideo(e.target.files[0]);
// });

// // SHOW VIDEO PREVIEW
// function handleVideo(file) {
//     if (!file) return;

//     const reader = new FileReader();

//     reader.onload = () => {
//         videoPreview.src = reader.result;
//         videoPreviewContainer.style.display = "block";
//     };

//     reader.readAsDataURL(file);
// }



document.addEventListener("DOMContentLoaded", function () {

    const startBtn = document.getElementById('startDetectionBtn');
    const uploadForm = document.getElementById('uploadForm');
    const summarySection = document.getElementById('summarySection');
    const accordionSection = document.getElementById('accordionSection');

    // hide accordion initially
    accordionSection.style.display = "none";

    // start detection button
    startBtn.addEventListener('click', () => {
        summarySection.style.display = 'none';
        uploadForm.style.display = 'block';
    });

    // detect defects button (submit)
    const detectBtn = document.querySelector(".detect-btn");

    detectBtn.addEventListener("click", function () {
        // DO NOT prevent submit
        accordionSection.style.display = "block";
    });

});

// Show Guide Video Section When Start Detection is Pressed
    document.getElementById("startDetectionBtn").addEventListener("click", function () {
        document.getElementById("guideVideoSection").style.display = "block";
    });

    // Play Button Logic
    document.getElementById("playGuideBtn").addEventListener("click", function () {
        const video = document.getElementById("guideVideo");

        video.classList.remove("d-none");
        video.play();

        this.style.display = "none"; // Hide play button after starting
    });



