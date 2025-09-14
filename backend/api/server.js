const { Path } = require("../../frontend/src/utils/enums");
const express = require("express");
const helmet = require("helmet");
const cors = require("cors");
const User = require("./user-model");
const { PassThrough } = require("stream");
const { analyzeVerbal } = require("./services/verbal_aa");
const { verbalPrompt } = require("./services/promptVerbal");
const { negativeFeedbackKeywords } = require("../constants");

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
  let prompt = [];
  prompt.push(
    `You are an interviewer assessing a candidate's response to a behavioural interview question highlighting the following topics: ${topics.join(
      ", "
    )}. Your response MUST HIGHLIGHT the relevance of the response to the question and must be given in the following format: "<sentence between 50 and 125 characters describing ONE relevance strength>;<sentence between 50 and 125 characters describing ONE relevance weakness>;<a number between 0 & 100 scoring relevance, no words/explanation>"`
  );
  prompt.push(`The question is: ${question} This is the response: ${response}`);
  return prompt;
};

const generateStructureClarityPrompt = (response, question, topics) => {
  let prompt = [];
  prompt.push(
    `You are an interviewer assessing a candidate's response to a behavioural interview question covering the following topics: ${topics.join(
      ", "
    )}. Your response MUST HIGHLIGHT their STARR structure & clarity and must be given in the following format: "<sentence between 50 and 125 characters describing ONE STARR structure/clarity strength>;<sentence between 50 and 125 characters describing ONE STARR structure/clarity weakness>;<a number between 0 & 100 scoring STARR structure/clarity, no words/explanation>"`
  );
  prompt.push(`The question is: ${question} This is the response: ${response}`);
  return prompt;
};

const generateInsightsPrompt = (response, question, topics) => {
  let prompt = [];
  prompt.push(
    `You are an interviewer assessing a candidate's response to a behavioural interview question covering the following topics: ${topics.join(
      ", "
    )}. Your response MUST HIGHLIGHT the quality & depth of insights and must be given in the following format: "<sentence between 50 and 125 characters describing ONE insight quality/depth strength>;<sentence between 50 and 125 characters describing ONE insight quality/depth weakness>;<a number between 0 & 100 scoring insight quality/depth, no words/explanation>"`
  );
  prompt.push(`The question is: ${question} This is the response: ${response}`);
  return prompt;
};

const generateVocabPrompt = (response, question, topics) => {
  let prompt = [];
  prompt.push(
    `You are an interviewer assessing a candidate's response to a behavioural interview question covering the following topics: ${topics.join(
      ", "
    )}. Your response MUST HIGHLIGHT the vocabulary & filler words used and must be given in the following format: "<sentence between 50 and 125 characters describing ONE vocab/filler words strength>;<sentence between 50 and 125 characters describing ONE vocab/filler words weakness>;<a number between 0 & 100 scoring vocab/filler words, no words/explanation>"`
  );
  prompt.push(`The question is: ${question} This is the response: ${response}`);
  return prompt;
};

const isLengthBetween50And125 = (text) => {
  const len = text.length;
  return len >= 50 && len <= 125;
};

const containsNegativeFeedback = (sentence) => {
  const lowerSentence = sentence.toLowerCase();

  // check each keyword
  return negativeFeedbackKeywords.some((keyword) =>
    lowerSentence.includes(keyword)
  );
};

const validateAndCleanseResponse = (response) => {
  const clean = response.split(";");
  const positivePeriods = clean[0].match(/\./g);
  const negativePeriods = clean[1].match(/\./g);

  if (
    clean.length - 1 !== 2 ||
    !clean[2].trim().match(/^(100|[1-9]?\d)$/) ||
    (positivePeriods && positivePeriods.length > 1) ||
    (negativePeriods && negativePeriods.length > 1) ||
    containsNegativeFeedback(clean[0]) ||
    !isLengthBetween50And125(clean[0]) ||
    !isLengthBetween50And125(clean[1])
  ) {
    return null;
  }

  let positiveFeedback = clean[0].replace(/^\s+|\s+$|\s+(?=\s)/g, "").trim();
  let negativeFeedback = clean[1].replace(/^\s+|\s+$|\s+(?=\s)/g, "").trim();
  positiveFeedback = positiveFeedback.replace(/\.$/, "");
  positiveFeedback =
    positiveFeedback.charAt(0).toUpperCase() +
    positiveFeedback.slice(1).toLowerCase();
  negativeFeedback = negativeFeedback.replace(/\.$/, "");
  negativeFeedback =
    negativeFeedback.charAt(0).toUpperCase() +
    negativeFeedback.slice(1).toLowerCase();

  if (positiveFeedback.length >= 100 || negativeFeedback.length >= 100)
    return null;

  return {
    positive: positiveFeedback,
    negative: negativeFeedback,
    score: clean[2].trim(),
  };
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

const computeFillerScore = (fillerCount, totalWords) => {
  const fillerRatio = fillerCount / totalWords;
  const minScore = 30;
  const maxScore = 100;
  const k = 10;

  const score = minScore + (maxScore - minScore) * Math.exp(-k * fillerRatio);
  return Math.round(score * 10) / 10; // 1 decimal
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

  let verbalScore = 0;
  let posVerbalFeedback = [];
  let negVerbalFeedback = [];
  let posNonverbalFeedback = [];
  let negNonverbalFeedback = [];
  let nonverbalScore = 0;
  let totalScore;
  let verbalScores = {};
  let nonVerbalScores = {};

  let relevanceResponse = null;
  let structureClarityResponse = null;
  let insightsResponse = null;
  let vocabResponse = null;
  let cleansedText = null;

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

    cleansedText = cleanseText(transcription.text);
    const noFillersText = removeFillerWords(cleansedText);

    const relevancePrompt = generateRelevancePrompt(
      noFillersText,
      req?.body?.question,
      req?.body?.topics
    );

    const structureClarityPrompt = generateStructureClarityPrompt(
      noFillersText,
      req?.body?.question,
      req?.body?.topics
    );

    const insightsPrompt = generateInsightsPrompt(
      noFillersText,
      req?.body?.question,
      req?.body?.topics
    );

    const vocabPrompt = generateVocabPrompt(
      cleansedText,
      req?.body?.question,
      req?.body?.topics
    );

    while (
      !relevanceResponse ||
      !structureClarityResponse ||
      !insightsResponse ||
      !vocabResponse
    ) {
      let clean;

      if (!relevanceResponse) {
        const newRelevanceResponse = await verbalPrompt(relevancePrompt);
        clean = validateAndCleanseResponse(newRelevanceResponse);
        if (clean) relevanceResponse = clean;
      }

      if (!structureClarityResponse) {
        const newStructureClarityResponse = await verbalPrompt(
          structureClarityPrompt
        );
        clean = validateAndCleanseResponse(newStructureClarityResponse);
        if (clean) structureClarityResponse = clean;
      }

      if (!insightsResponse) {
        const newInsightsResponse = await verbalPrompt(insightsPrompt);
        clean = validateAndCleanseResponse(newInsightsResponse);
        if (clean) insightsResponse = clean;
      }

      if (!vocabResponse) {
        const newVocabResponse = await verbalPrompt(vocabPrompt);
        clean = validateAndCleanseResponse(newVocabResponse);
        if (clean) vocabResponse = clean;
      }
    }

    // Make sure to do validation to ensure proper format, otherwise tell it to redo
  } catch (error) {
    next(error);
    return;
  }

  const vocabAndFillersResponse = await fetch(
    `http://127.0.0.1:8000/vocab-fillers`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ cleansedText }),
    }
  );

  let vocabScore, fillerWordScore;

  if (vocabAndFillersResponse.ok) {
    const verbalAnalysis = await vocabAndFillersResponse.json();
    vocabScore = verbalAnalysis[0];
    fillerWordScore = computeFillerScore(verbalAnalysis[1], verbalAnalysis[2]);
  } else {
    next({ status: 404, message: "Something went wrong" });
  }

  verbalScore +=
    parseInt(relevanceResponse.score) +
    parseInt(structureClarityResponse.score) +
    parseInt(insightsResponse.score) +
    vocabScore +
    fillerWordScore;
  verbalScore /= 5;
  verbalScore = verbalScore.toFixed(1);
  verbalScores.relevanceScore = relevanceResponse.score;
  verbalScores.structureClarityScore = structureClarityResponse.score;
  verbalScores.insightsScore = insightsResponse.score;
  verbalScores.vocabScore = vocabScore;
  verbalScores.fillerWordScore = fillerWordScore;

  posVerbalFeedback.push(relevanceResponse.positive);
  posVerbalFeedback.push(structureClarityResponse.positive);
  posVerbalFeedback.push(insightsResponse.positive);
  posVerbalFeedback.push(vocabResponse.positive);

  negVerbalFeedback.push(relevanceResponse.negative);
  negVerbalFeedback.push(structureClarityResponse.negative);
  negVerbalFeedback.push(insightsResponse.negative);
  negVerbalFeedback.push(vocabResponse.negative);

  // NON-VERBAL ANALYSIS
  const nonverbalResponse = await fetch(`http://127.0.0.1:8000/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ videoUrl: req?.body?.videoUrl }),
  });

  totalScore = (verbalScore + nonverbalScore) / 2;

  await User.storePracticeVideoResults(
    parseInt(req?.body?.practiceRunId),
    posVerbalFeedback,
    negVerbalFeedback,
    posNonverbalFeedback,
    negNonverbalFeedback,
    req?.body?.videoUrl,
    parseInt(req?.query?.userId),
    nonVerbalScores,
    verbalScores,
    verbalScore,
    nonverbalScore,
    totalScore
  );

  res.status(200).end();
});

server.use("/", async (req, res, next) => {
  next({ status: 404, message: "Endpoint not found" });
});

server.use((err, req, res, next) => {
  const { message, status = 500 } = err;
  res.status(status).json({ message });
});

module.exports = server;
