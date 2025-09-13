const { Path } = require("../../frontend/src/utils/enums");
const express = require("express");
const helmet = require("helmet");
const cors = require("cors");
const User = require("./user-model");
const { PassThrough } = require("stream");
const { analyzeVerbal } = require("./services/verbal");

const server = express();
server.use(helmet());
server.use(express.json());
const corsOptions = {
  origin: process.env.FRONTEND_URL,
};
server.use(cors(corsOptions));
server.options("/", cors(corsOptions));

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
  // try {
  //   const transcription = await analyzeVerbal(testAudio);
  //   console.log(transcription);
  // } catch (error) {
  //   next(error);
  //   return;
  // }

  // NON-VERBAL ANALYSIS
  const nonverbalResponse = await fetch(
    `http://127.0.0.1:8000/`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ videoUrl: req?.body?.videoUrl }),
    }
  );

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
