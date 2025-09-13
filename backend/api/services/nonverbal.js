// make sure to run "brew install ffmpeg" on your local

const { HandLandmarker, FilesetResolver } = require("@mediapipe/tasks-vision");
const { createCanvas, loadImage, Image } = require("canvas");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("ffmpeg-static");
const ffprobePath = require("ffprobe-static").path;
const fs = require("fs");
const path = require("path");

// Specify the path of the executable file
ffmpeg.setFfmpegPath(ffmpegPath);
ffmpeg.setFfprobePath(ffprobePath);

/**
 * Extract frames from the video and detect key points of the hand
 * @param {string} videoPath video file path
 * @param {number} frameRate how many frame per sec
 * @returns {Promise<Array>} The detection result of each frame
 */
async function detectHandLandmarksFromVideo(videoPath, frameRate = 1) {
    // Extract the frame to the temporary directory
    const framesDir = path.join(__dirname, "frames_tmp");
    if (!fs.existsSync(framesDir)) fs.mkdirSync(framesDir);

    await new Promise((resolve, reject) => {
    ffmpeg(videoPath)
        .outputOptions([`-vf fps=${frameRate}`])
        .output(path.join(framesDir, "frame-%03d.png"))
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    // load model
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
        modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
        },
        runningMode: "IMAGE",
        numHands: 2,
    });

    // Perform detection on each frame
    const results = [];
    const frameFiles = fs.readdirSync(framesDir).filter(f => f.endsWith(".png"));
    for (const file of frameFiles) {
        const framePath = path.join(framesDir, file);
        const image = await loadImage(framePath);
        const canvas = createCanvas(image.width, image.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(image, 0, 0);
        const result = handLandmarker.detect(canvas);
        results.push({ frame: file, result });
    }

    // Clear temporary frames
    frameFiles.forEach(f => fs.unlinkSync(path.join(framesDir, f)));
    fs.rmdirSync(framesDir);

    return results;
}

module.exports = {
    detectHandLandmarksFromVideo,
};


// testing
if (require.main === module) {
  const testVideo = path.join(__dirname, "../../../test/test.mp4");
  detectHandLandmarksFromVideo(testVideo, 1) // one frame per second
    .then(results => {
      console.log("Hand landmarks results:", JSON.stringify(results, null, 2));
    })
    .catch(err => {
      console.error("Error:", err);
    });
}
