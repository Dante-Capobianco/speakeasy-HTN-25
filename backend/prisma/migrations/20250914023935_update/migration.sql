/*
  Warnings:

  - The `totalScore` column on the `PracticeRun` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - The `verbalScore` column on the `PracticeRun` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - The `nonVerbalScore` column on the `PracticeRun` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- AlterTable
ALTER TABLE "public"."PracticeRun" ADD COLUMN     "eyeMovementsScore" INTEGER[],
ADD COLUMN     "facialExpressionScore" INTEGER[],
ADD COLUMN     "fillerWordScore" INTEGER[],
ADD COLUMN     "handGesturesScore" INTEGER[],
ADD COLUMN     "insightsScore" INTEGER[],
ADD COLUMN     "pausingScore" INTEGER[],
ADD COLUMN     "postureScore" INTEGER[],
ADD COLUMN     "relevanceScore" INTEGER[],
ADD COLUMN     "spatialDistributionScore" INTEGER[],
ADD COLUMN     "structureClarityScore" INTEGER[],
ADD COLUMN     "vocabScore" INTEGER[],
DROP COLUMN "totalScore",
ADD COLUMN     "totalScore" INTEGER[],
DROP COLUMN "verbalScore",
ADD COLUMN     "verbalScore" INTEGER[],
DROP COLUMN "nonVerbalScore",
ADD COLUMN     "nonVerbalScore" INTEGER[];
