const fs = require("fs");
require('dotenv').config();
const OpenAI = require("openai");

const openai = new OpenAI(); // 自动读取 OPENAI_API_KEY


/**
 * OpenAI Whisper API audio to text
 * @param {string} audioPath audio file path: // TODO changed later
 * @returns {Promise<string>} returned text
 */
async function analyzeVerbal(audioPath) {
        const transcription = await openai.audio.transcriptions.create({
        file: fs.createReadStream(audioPath),
        model: "gpt-4o-transcribe", 
    });
    return transcription.text;
}

module.exports = {
    analyzeVerbal,
};


// For test 
if (require.main === module) {
    const path = require("path");
    const testAudio = path.join(__dirname, "../../../test/test.mp4");
    analyzeVerbal(testAudio)
        .then(text => {
        console.log("Transcription result:", text);
        })
        .catch(err => {
        console.error("Error:", err);
        });
}