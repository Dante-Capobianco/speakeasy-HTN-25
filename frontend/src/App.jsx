import "./App.css";
import { uploadVideoAndGetLink } from "./utils/helperFunctions";
import { useState } from "react";
import { getUser } from "./utils/helperFunctions";

function App() {
  const [videoUrl, setVideoUrl] = useState(null);

  return (
    <>
      <div>
        This is root page of application; index.css is for styling to apply
        across entire website; app.css for page-specific styling
      </div>
      <button onClick={() => getUser(1)}>get User</button>
      <input
        type="file"
        id="practice-video"
        accept="video/*"
        onChange={(e) => setVideoUrl(uploadVideoAndGetLink(e.target))}
      />
    </>
  );
}

export default App;
