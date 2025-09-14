const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

module.exports = {
  async findUserById(userId) {
    return await prisma.user.findUnique({
      where: {
        id: userId,
      },
    });
  },

  async addUser(topics) {
    const newUser = await prisma.user.create({
      data: {
        topics,
      },
    });

    return newUser.id;
  },

  async createPracticeRun(
    userId,
    topics,
    questions,
    timeToReadQuestion,
    timeToAnswerQuestion
  ) {
    const practiceRun = await prisma.practiceRun.create({
      data: {
        userId,
        topics,
        questions,
        timeToReadQuestion,
        timeToAnswerQuestion,
        posVerbalFeedback: [],
        negVerbalFeedback: [],
        posNonverbalFeedback: [],
        negNonverbalFeedback: [],
        videos: [],

        nonVerbalScore: [],
        facialExpressionScore: [],
        eyeMovementsScore: [],
        pausingScore: [],
        postureScore: [],
        handGesturesScore: [],
        spatialDistributionScore: [],

        verbalScore: [],
        relevanceScore: [],
        structureClarityScore: [],
        insightsScore: [],
        vocabScore: [],
        fillerWordScore: [],

        totalScore: [],
      },
    });

    return practiceRun.id;
  },

  async storePracticeVideoResults(
    practiceRunId,
    posVerbalFeedback,
    negVerbalFeedback,
    posNonverbalFeedback,
    negNonverbalFeedback,
    videoUrl,
    nonverbalScores,
    verbalScores,
    verbalScore,
    nonverbalScore,
    totalScore
  ) {
    await prisma.practiceRun.update({
      where: { id: practiceRunId },
      data: {
        posVerbalFeedback: { push: JSON.stringify(posVerbalFeedback) },
        negVerbalFeedback: { push: JSON.stringify(negVerbalFeedback) },
        posNonverbalFeedback: { push: JSON.stringify(posNonverbalFeedback) },
        negNonverbalFeedback: { push: JSON.stringify(negNonverbalFeedback) },
        videos: { push: videoUrl },

        nonVerbalScore: { push: nonverbalScore },
        facialExpressionScore: {
          push: nonverbalScores.facialExpressionScore,
        },
        eyeMovementsScore: { push: nonverbalScores.eyeMovementsScore },
        pausingScore: { push: nonverbalScores.pausingScore },
        postureScore: { push: nonverbalScores.postureScore },
        handGesturesScore: { push: nonverbalScores.handGesturesScore },
        spatialDistributionScore: {
          push: nonverbalScores.spatialDistributionScore,
        },

        verbalScore: { push: verbalScore },
        relevanceScore: { push: verbalScores.relevanceScore },
        structureClarityScore: {
          push: verbalScores.structureClarityScore,
        },
        insightsScore: { push: verbalScores.insightsScore },
        vocabScore: { push: verbalScores.vocabScore },
        fillerWordScore: { push: verbalScores.fillerWordScore },

        totalScore: { push: totalScore },
      },
    });
  },
};
