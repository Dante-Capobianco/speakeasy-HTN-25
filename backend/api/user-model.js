const { PrismaClient } = require("@prisma/client");
const prisma = new PrismaClient();

module.exports = {
  async findAllUsers() {
    return await prisma.user.findMany({});
  },
};
