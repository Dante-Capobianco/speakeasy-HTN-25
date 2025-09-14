import "./App.css";
import { processVideo } from "./utils/helperFunctions";
import { getUser, addUser } from "./utils/helperFunctions";

function App() {

  return (
    <>
      <div>
        This is root page of application; index.css is for styling to apply
        across entire website; app.css for page-specific styling
      </div>
      <button onClick={() => getUser(1)}>get User</button>
      <button onClick={() => addUser(["Communication, Leadership"])}>Add User</button>
      <button onClick={() => console.log(generateQuestions(2, ["Communication, Leadership"]))}>Generate Q's</button>
      <input
        type="file"
        id="practice-video"
        accept="video/*"
        onChange={(e) => processVideo(e.target, "Tell me a bit about yourself?", 1, ["Work Ethic, Leadership"])}
      />
    </>
  );
}

export default App;
