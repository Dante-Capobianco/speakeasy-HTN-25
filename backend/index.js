import dotenv from "dotenv";
import server from "./api/server.js"; // note the .js extension

dotenv.config();

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});