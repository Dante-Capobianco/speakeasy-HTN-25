/*
  Warnings:

  - You are about to drop the column `negativeFeedback` on the `PracticeRun` table. All the data in the column will be lost.
  - You are about to drop the column `positiveFeedback` on the `PracticeRun` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "public"."PracticeRun" DROP COLUMN "negativeFeedback",
DROP COLUMN "positiveFeedback",
ADD COLUMN     "negNonverbalFeedback" TEXT[],
ADD COLUMN     "negVerbalFeedback" TEXT[],
ADD COLUMN     "posNonverbalFeedback" TEXT[],
ADD COLUMN     "posVerbalFeedback" TEXT[];
