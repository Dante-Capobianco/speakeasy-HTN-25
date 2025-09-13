require('dotenv').config();
const axios = require('axios');
const fs = require('fs-extra');
const path = require('path');

const BASE_URL = 'https://api.assemblyai.com/v2';
const API_KEY = process.env.ASSEMBLYAI_API_KEY;

if (!API_KEY && require.main === module) {
  console.warn('Warning: ASSEMBLYAI_API_KEY is missing in backend/.env');
}

const client = axios.create({
  baseURL: BASE_URL,
  headers: { authorization: API_KEY },
});

/**
 * If the file path is local, it will be uploaded to AssemblyAI first. If it is http(s), it will be returned directly
 * We will implement cloud later
 * @param {string} inputPathOrUrl
 * @returns {Promise<string>} Audio urls available for transcription
 */
async function getAudioUrl(inputPathOrUrl) {
  const isUrl = /^https?:\/\//i.test(inputPathOrUrl);
  if (isUrl) return inputPathOrUrl;

  const data = await fs.readFile(inputPathOrUrl);
  const res = await client.post('/upload', data);
  return res.data.upload_url;
}

/**
 * Transcribe audio/video with AssemblyAI, retaining padding by default (disfluencies: true)
 * @param {string} inputPathOrUrl Local file path or remote URL (mp3/mp4, etc.)
 * @param {object} options Optional { speech_model, disfluencies, punctuate, pollIntervalMs, timeoutMs }
 * @returns {Promise<{ text: string, id: string, full: object }>}
 */
async function analyzeVerbal(inputPathOrUrl, options = {}) {
  const audio_url = await getAudioUrl(inputPathOrUrl);

  const payload = {
    audio_url,
    speech_model: options.speech_model || 'universal',
    // Keep fillers like "uh/um"
    disfluencies: options.disfluencies !== undefined ? options.disfluencies : true,
    punctuate: options.punctuate !== undefined ? options.punctuate : true,
  };

  const { data } = await client.post('/transcript', payload);
  const id = data.id;
  const pollingPath = `/transcript/${id}`;

  const start = Date.now();
  const pollInterval = options.pollIntervalMs || 3000;
  const timeout = options.timeoutMs || 10 * 60 * 1000; // 10 分钟超时

  // Polling to end
  while (true) {
    await new Promise((r) => setTimeout(r, pollInterval));
    const poll = await client.get(pollingPath);
    const st = poll.data.status;

    if (st === 'completed') {
      return { text: poll.data.text, id, full: poll.data };
    }
    if (st === 'error') {
      const err = new Error(`Transcription failed: ${poll.data.error}`);
      err.details = poll.data;
      throw err;
    }
    if (Date.now() - start > timeout) {
      throw new Error('Transcription timed out');
    }
  }
}

module.exports = { analyzeVerbal };

// For test: node api/services/verbal.js
if (require.main === module) {
  const testAudio = path.join(__dirname, '../../../test/test(2).mp4'); // root test/test.mp4
  analyzeVerbal(testAudio, { pollIntervalMs: 3000, disfluencies: true })
    .then((res) => {
      console.log('Transcript:\n', res.text);
    })
    .catch((err) => {
      console.error('Error:', err.message);
      if (err.details) console.error(err.details);
    });
}