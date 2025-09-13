const express = require("express");
const helmet = require("helmet");
const cors = require("cors");

const server = express();
server.use(helmet());
server.use(express.json());
const corsOptions = {
  origin: process.env.FRONTEND_URL
};
server.use(cors(corsOptions));
server.options("/", cors(corsOptions));

server.use("/", (req, res, next) => {
  next({ status: 404, message: "Endpoint not found" });
});

server.use((err, req, res, next) => {
  const { message, status = 500 } = err;
  res.status(status).json({ message });
});

module.exports = server;