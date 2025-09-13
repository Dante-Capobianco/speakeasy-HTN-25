-- CreateTable
CREATE TABLE "public"."User" (
    "id" SERIAL NOT NULL,
    "topics" TEXT[],

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."PracticeRun" (
    "id" SERIAL NOT NULL,
    "topics" TEXT[],
    "questions" TEXT[],
    "timeToReadQuestion" INTEGER NOT NULL,
    "timeToAnswerQuestion" INTEGER NOT NULL,
    "totalScore" INTEGER NOT NULL,
    "verbalScore" INTEGER NOT NULL,
    "nonVerbalScore" INTEGER NOT NULL,
    "positiveFeedback" TEXT[],
    "negativeFeedback" TEXT[],
    "userId" INTEGER NOT NULL,
    "videos" TEXT[],

    CONSTRAINT "PracticeRun_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "public"."PracticeRun" ADD CONSTRAINT "PracticeRun_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
