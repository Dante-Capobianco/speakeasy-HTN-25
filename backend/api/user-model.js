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
};
