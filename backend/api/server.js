const { Path } = require("../../frontend/src/utils/enums");
const express = require("express");
const helmet = require("helmet");
const cors = require("cors");
const User = require("./user-model");
const { PassThrough } = require("stream");
const { analyzeVerbal } = require("./services/verbal_aa");

const server = express();
server.use(helmet());
server.use(express.json());
const corsOptions = {
  origin: process.env.FRONTEND_URL,
};
server.use(cors(corsOptions));
server.options("/", cors(corsOptions));

// Prompts
const generateRelevancePrompt = (response, question, topics) => {
  return `You are an interviewer assessing a candidate's response based on its relevance to a behavioural interview question highlighting the following topics: ${topics.join(
    ", "
  )}. The question is: ${question} This is the response: ${response} Given this, provide feedback in the following format: <1 sentence describing relevance strength>;<1 sentence describing relevance weakness>;<relevance score 0 to 100>`;
};

const cleanseText = (text) => {
  let cleansedText = text.toLowerCase();
  cleansedText = cleansedText.replace(/^\s+|\s+$|\s+(?=\s)/g, "").trim();

  return cleansedText;
};

const removeFillerWords = (text) => {
  const fillerWords = [
    "um",
    "uh",
    "like",
    "and yeah",
    "so yeah",
    "you know",
    "so",
    "actually",
    "basically",
    "right",
    "i mean",
    "okay",
    "well",
    "yeah",
  ];
  const regex = new RegExp(`\\b(${fillerWords.join("|")})\\b,?`, "gi");

  const cleaned = text.replace(regex, "").replace(/\s+/g, " ").trim();
  return cleaned;
};

server.get(Path.GET_USER, async (req, res, next) => {
  const user = await User.findUserById(parseInt(req?.query?.id));
  res.status(200).json({ user });
});

server.post(Path.ADD_USER, async (req, res, next) => {
  const userId = await User.addUser(req?.body?.topics);
  res.status(200).json({ userId });
});

server.post(Path.ANALYZE_VIDEO, async (req, res, next) => {
  const videoFile = await fetch(req?.body?.videoUrl);
  if (!videoFile.ok) {
    next({ status: 404, message: "Failed to fetch file" });
    return;
  }

  // Convert response to a stream
  // const buffer = await videoFile.arrayBuffer();
  // const stream = new PassThrough();
  // stream.end(Buffer.from(buffer));

  // VERBAL ANALYSIS
  try {
    const transcription = await analyzeVerbal(req?.body?.videoUrl, {
      pollIntervalMs: 3000,
      disfluencies: true,
    });

    if (!transcription.text) {
      next({ status: 404, message: "Something went wrong" });
      return;
    }

    const cleansedText = cleanseText(transcription.text);
    const noFillersText = removeFillerWords(cleansedText);

    const relevancePrompt = generateRelevancePrompt(
      noFillersText,
      req?.body?.question,
      req?.body?.topics
    );

    // Make sure to do validation to ensure proper format, otherwise tell it to redo
  } catch (error) {
    next(error);
    return;
  }

  // NON-VERBAL ANALYSIS
  const nonverbalResponse = await fetch(`http://127.0.0.1:8000/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ videoUrl: req?.body?.videoUrl }),
  });

  const analysis = null; // await User.addVideoAnalysis(null, parseInt(req?.query?.id));
  res.status(200).json({ analysis });
});

server.use("/", async (req, res, next) => {
  next({ status: 404, message: "Endpoint not found" });
});

server.use((err, req, res, next) => {
  const { message, status = 500 } = err;
  res.status(status).json({ message });
});

module.exports = server;
