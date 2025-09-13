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
};
