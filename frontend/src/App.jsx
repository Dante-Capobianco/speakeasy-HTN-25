import "./App.css";
import { Path } from "./utils/enums";

function App() {
  const getUser = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BASE_URL}${Path.GET_USER}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          }
        }
      );

      if (response.ok) {
        // Do something
      } else {
        // Do something
      }
    } catch (error) {
      // Do something
    }
  };

  return (
    <>
      <div>
        This is root page of application; index.css is for styling to apply
        across entire website; app.css for page-specific styling
      </div>
      <button onClick={getUser}>Get User</button>
    </>
  );
}

export default App;
