import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import { addUser, generateQuestions, getUser } from "./utils/helperFunctions";
import { Path } from "./utils/enums";

function App() {
  // --- Landing Page ---
  const fullText = "Welcome to SpeakEasy";
  const [displayedText, setDisplayedText] = useState("");
  const [page, setPage] = useState("landing");
  const [userTopics, setUserTopics] = useState([]);

  // --- States for topics and practice setup ---
  const [selectedTopics, setSelectedTopics] = useState([]);
  const [showTopicsDropdown, setShowTopicsDropdown] = useState(false);
  const [showQtyDropdown, setShowQtyDropdown] = useState(false);
  const [showPrepDropdown, setShowPrepDropdown] = useState(false);
  const [showAnswerDropdown, setShowAnswerDropdown] = useState(false);
  const [practiceTopics, setPracticeTopics] = useState([]);
  const [questionQty, setQuestionQty] = useState("Choose question qty");
  const [prepTime, setPrepTime] = useState("Choose prep time");
  const [answerTime, setAnswerTime] = useState("Choose answer time");
  const [automaticMode, setAutomaticMode] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [prepTimeLeft, setPrepTimeLeft] = useState(0);
  const [isPreparing, setIsPreparing] = useState(false);
  const [totalQuestions, setTotalQuestions] = useState(1);

  // --- Video Recording States ---
  const [answerTimeLeft, setAnswerTimeLeft] = useState(0);
  const [isAnswering, setIsAnswering] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [stream, setStream] = useState(null);
  const [cameraError, setCameraError] = useState(null);

  // --- Integration states ---
  const [userId, setUserId] = useState(null);
  const [user, setUser] = useState(null);
  const [currPracRunId, setCurrPracRunId] = useState(null);
  const [questions, setQuestions] = useState(null);

  const videoRef = useRef(null);
  const recordedVideoRef = useRef(null);

  const dropdownTopics = [
    "Teamwork",
    "Problem Solving",
    "Leadership/Initiative",
    "Adaptability",
    "Communication",
    "Project Management",
  ];

  // --- Typing Effect for Landing Page ---
  useEffect(() => {
    const getUserObj = async () => {
      const user = await getUser(userId);
      setUser(user);
    };

    if (page === "landing") {
      let index = 0;
      const interval = setInterval(() => {
        if (index <= fullText.length) {
          setDisplayedText(fullText.slice(0, index));
          index++;
        } else {
          clearInterval(interval);
        }
      }, 150);
      return () => clearInterval(interval);
    } else if (page === "practice") {
      getUserObj();
    }
  }, [page]);

  // --- Prep Timer Effect ---
  useEffect(() => {
    let timer;
    if (isPreparing && prepTimeLeft > 0) {
      timer = setInterval(() => {
        setPrepTimeLeft((prev) => {
          if (prev <= 1) {
            setIsPreparing(false);
            setPage("answer");
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isPreparing, prepTimeLeft]);

  // NEWLY ADDED FROM CHATGPT
  useEffect(() => {
    if (page !== "prepare") return;

    const prepTimeMap = {
      "15 s": 15,
      "30 s": 30,
      "45 s": 45,
      "1 min": 60,
      "1min 15 sec": 75,
    };
    const timeInSeconds = prepTimeMap[prepTime] || 30;

    setPrepTimeLeft(timeInSeconds);
    setIsPreparing(true);
  }, [page, currentQuestion, prepTime]);

  // --- Answer Timer Effect ---
  useEffect(() => {
    let timer;
    if (isAnswering && answerTimeLeft > 0 && page === "answer") {
      timer = setInterval(() => {
        setAnswerTimeLeft((prev) => {
          if (prev <= 1) {
            // Auto-stop recording when time is up
            stopRecording();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isAnswering, answerTimeLeft, page]);

  // --- Camera Setup Effect ---
  useEffect(() => {
    if (page === "answer") {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [page]);

  // --- Camera Functions ---
  const startCamera = async () => {
    try {
      setCameraError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true,
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      setCameraError("Unable to access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const startRecording = () => {
    if (!stream) {
      setCameraError("Camera not ready. Please wait and try again.");
      return;
    }

    try {
      // Reset any existing recording state
      setRecordedBlob(null);
      setIsRecording(false);
      setIsAnswering(false);

      const recorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp9",
      });

      const chunks = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        setRecordedBlob(blob);
        setIsRecording(false);
        setIsAnswering(false);
      };

      // Start recording
      setMediaRecorder(recorder);
      recorder.start();
      setIsRecording(true);

      // Start answer countdown timer
      const answerTimeMap = {
        "1 min 30 s": 90,
        "2 min": 120,
        "2 min 30 s": 150,
        "3 min": 180,
        "5 min": 300,
      };
      const timeInSeconds = answerTimeMap[answerTime] || 120;
      setAnswerTimeLeft(timeInSeconds);
      setIsAnswering(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setCameraError("Unable to start recording. Please try again.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
    }
  };

  // --- Handlers ---
  const toggleTopic = (topic) => {
    if (selectedTopics.includes(topic)) {
      setSelectedTopics(selectedTopics.filter((t) => t !== topic));
    } else if (selectedTopics.length < 3) {
      setSelectedTopics([...selectedTopics, topic]);
    }
  };

  const handleContinue = async () => {
    try {
      const id = await addUser(selectedTopics);
      setUserId(id);
      if (selectedTopics.length === 3) {
        setUserTopics(selectedTopics);
        setPage("summary");
      }
    } catch (err) {}
  };

  const addPracticeTopic = (topic) => {
    if (!practiceTopics.includes(topic) && practiceTopics.length < 3) {
      setPracticeTopics([...practiceTopics, topic]);
    }
    setShowTopicsDropdown(false);
  };

  const removePracticeTopic = (topic) => {
    setPracticeTopics(practiceTopics.filter((t) => t !== topic));
  };

  const startPractice = async () => {
    const numQ = parseInt(questionQty, 10) || 1;

    try {
      const questions = await generateQuestions(numQ, practiceTopics);

      const response = await fetch(
        `${import.meta.env.VITE_BASE_URL}${
          Path.CREATE_PRACTICE_RUN
        }?id=${userId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            practiceTopics,
            questions,
            prepTime,
            answerTime,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        setCurrPracRunId(data.id);
        console.log(data.id, userId, questions)
        setQuestions(questions)
        
        setTotalQuestions(numQ);
        setCurrentQuestion(1);

        // Reset recording-related state
        setRecordedBlob(null);
        setIsAnswering(false);
        setIsRecording(false);
        setAnswerTimeLeft(0);

        setPage("prepare");
      }
    } catch (error) {}
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const handleAnswerComplete = () => {
    stopRecording();
    setPage("completed");
  };

  // --- Header Component ---
  const Header = () => (
    <div className="header">
      <button onClick={() => setPage("landing")} className="header-button">
        SpeakEasy
      </button>
    </div>
  );

  // --- Render Pages ---
  if (page === "landing") {
    return (
      <div className="landing-page">
        <h1 className="landing-title">
          {displayedText}
          <span className="cursor">|</span>
        </h1>
        {displayedText === fullText && (
          <button onClick={() => setPage("topics")} className="continue-button">
            Continue
          </button>
        )}
      </div>
    );
  }

  if (page === "topics") {
    const topics = [
      "Teamwork",
      "Problem Solving",
      "Leadership/Initiative",
      "Adaptability",
      "Communication",
      "Project Management",
    ];

    return (
      <div className="topics-page">
        <Header />
        <h2 className="topics-title">Choose up to 3 topics</h2>
        <div className="topics-grid">
          {topics.map((topic) => {
            const isSelected = selectedTopics.includes(topic);
            return (
              <button
                key={topic}
                onClick={() => toggleTopic(topic)}
                className={`topic-button ${isSelected ? "selected" : ""}`}
              >
                {topic}
              </button>
            );
          })}
        </div>
        <button
          onClick={handleContinue}
          disabled={selectedTopics.length === 0}
          className={`continue-button ${
            selectedTopics.length === 0 ? "disabled" : ""
          }`}
        >
          Continue
        </button>
      </div>
    );
  }

  if (page === "summary") {
    const steps = [
      {
        title: "Choose what to practice",
        desc: "Pick the skills you want to focus on.",
      },
      {
        title: "Mimic real world 1 way interviews",
        desc: "Practice in a realistic environment.",
      },
      {
        title: "Gain insights, iterate, improve",
        desc: "Get feedback and refine your skills.",
      },
    ];

    return (
      <div className="summary-page">
        <Header />
        <h2 className="page-title">What happens next</h2>
        <div className="steps-grid">
          {steps.map((step) => (
            <div key={step.title} className="step-item">
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </div>
          ))}
        </div>
        <button onClick={() => setPage("practice")} className="continue-button">
          Start Practicing
        </button>
      </div>
    );
  }

  if (page === "practice") {
    const pastRuns = [
      { id: 1, label: "Practice Run 1" },
      { id: 2, label: "Practice Run 2" },
      { id: 3, label: "Practice Run 3" },
    ];

    return (
      <div className="practice-page">
        <Header />

        <div className="new-practice-section">
          <button
            onClick={() => setPage("newPractice")}
            className="plus-button"
          >
            +
          </button>
          <span className="new-practice-text">Start new practice</span>
        </div>

        <div className="past-runs-section">
          {user?.practiceRuns?.map((run) => (
            <div key={run.id} className="run-item">
              <span className="run-label">{run.label}</span>
              <button className="view-insights-button">View Insights</button>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (page === "newPractice") {
    return (
      <div className="new-practice-page">
        <Header />

        <h2 className="page-title">New Practice Run</h2>

        <div className="form-section">
          <h3 className="section-title">Topics</h3>
          <div className="dropdown-container">
            <button
              onClick={() => setShowTopicsDropdown(!showTopicsDropdown)}
              className="dropdown-button"
            >
              Select topics
              <span className="dropdown-arrow">▼</span>
            </button>

            {showTopicsDropdown && (
              <div className="dropdown-menu">
                {dropdownTopics.map((topic) => (
                  <button
                    key={topic}
                    onClick={() => addPracticeTopic(topic)}
                    className="dropdown-item"
                  >
                    {topic}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="selected-topics">
            {practiceTopics.map((topic) => (
              <div key={topic} className="selected-topic">
                {topic}
                <button
                  onClick={() => removePracticeTopic(topic)}
                  className="remove-topic"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          <div className="checkbox-container">
            <input
              type="checkbox"
              checked={automaticMode}
              onChange={(e) => setAutomaticMode(e.target.checked)}
            />
            <span className="checkbox-label">Automatic</span>
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">Number of questions</h3>
          <div className="dropdown-container">
            <button
              onClick={() => setShowQtyDropdown(!showQtyDropdown)}
              className="dropdown-button"
            >
              {questionQty}
              <span className="dropdown-arrow">▼</span>
            </button>

            {showQtyDropdown && (
              <div className="dropdown-menu">
                {["1", "2", "3"].map((qty) => (
                  <button
                    key={qty}
                    onClick={() => {
                      setQuestionQty(qty);
                      setShowQtyDropdown(false);
                    }}
                    className="dropdown-item"
                  >
                    {qty}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">Prep Time Per Question</h3>
          <div className="dropdown-container">
            <button
              onClick={() => setShowPrepDropdown(!showPrepDropdown)}
              className="dropdown-button"
            >
              {prepTime}
              <span className="dropdown-arrow">▼</span>
            </button>

            {showPrepDropdown && (
              <div className="dropdown-menu">
                {["15 s", "30 s", "45 s", "1 min", "1 min 15 sec"].map(
                  (time) => (
                    <button
                      key={time}
                      onClick={() => {
                        setPrepTime(time);
                        setShowPrepDropdown(false);
                      }}
                      className="dropdown-item"
                    >
                      {time}
                    </button>
                  )
                )}
              </div>
            )}
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">Answer time per question</h3>
          <div className="dropdown-container">
            <button
              onClick={() => setShowAnswerDropdown(!showAnswerDropdown)}
              className="dropdown-button"
            >
              {answerTime}
              <span className="dropdown-arrow">▼</span>
            </button>

            {showAnswerDropdown && (
              <div className="dropdown-menu">
                {["1 min 30 s", "2 min", "2 min 30 s", "3 min", "5 min"].map(
                  (time) => (
                    <button
                      key={time}
                      onClick={() => {
                        setAnswerTime(time);
                        setShowAnswerDropdown(false);
                      }}
                      className="dropdown-item"
                    >
                      {time}
                    </button>
                  )
                )}
              </div>
            )}
          </div>
        </div>

        <button onClick={startPractice} className="get-started-button">
          Get Started!
        </button>
      </div>
    );
  }

  if (page === "prepare") {
    const totalTime =
      prepTime === "15 s"
        ? 15
        : prepTime === "30 s"
        ? 30
        : prepTime === "45 s"
        ? 45
        : prepTime === "1 min"
        ? 60
        : 75;
    const progress = ((totalTime - prepTimeLeft) / totalTime) * 283; // 283 is circumference of circle with radius 45

    return (
      <div className="prepare-page">
        <Header />

        <h2 className="page-title">Prepare: Question {currentQuestion}</h2>
        <h3 className="question-text">
          Describe a time when you had to work as part of a team to solve a
          difficult problem.
        </h3>

        <div className="timer-container">
          <svg className="timer-svg">
            <circle className="timer-bg" cx="100" cy="100" r="45" />
            <circle
              className="timer-progress"
              cx="100"
              cy="100"
              r="45"
              style={{
                strokeDashoffset: 283 - progress,
              }}
            />
          </svg>
          <div className="timer-text">{formatTime(prepTimeLeft)}</div>
        </div>
      </div>
    );
  }

  if (page === "answer") {
    const answerTimeMap = {
      "1 min 30 s": 90,
      "2 min": 120,
      "2 min 30 s": 150,
      "3 min": 180,
      "5 min": 300,
    };
    const totalTime = answerTimeMap[answerTime] || 120;
    const progress = isAnswering
      ? ((totalTime - answerTimeLeft) / totalTime) * 283
      : 0;

    return (
      <div className="answer-page">
        <Header />

        <h2 className="page-title">Answer: Question {currentQuestion}</h2>
        <h3 className="question-text">
          Describe a time when you had to work as part of a team to solve a
          difficult problem.
        </h3>

        <div className="video-container">
          {cameraError ? (
            <div className="camera-error">
              <p>{cameraError}</p>
              <button onClick={startCamera} className="retry-button">
                Retry Camera Access
              </button>
            </div>
          ) : (
            <>
              <video ref={videoRef} autoPlay muted className="video-preview" />

              {recordedBlob && (
                <video
                  ref={recordedVideoRef}
                  src={URL.createObjectURL(recordedBlob)}
                  controls
                  className="recorded-video"
                />
              )}
            </>
          )}
        </div>

        {isAnswering && (
          <div className="timer-container">
            <svg className="timer-svg">
              <circle className="timer-bg" cx="100" cy="100" r="45" />
              <circle
                className="timer-progress"
                cx="100"
                cy="100"
                r="45"
                style={{
                  strokeDashoffset: 283 - progress,
                }}
              />
            </svg>
            <div className="timer-text">{formatTime(answerTimeLeft)}</div>
          </div>
        )}

        <div className="recording-controls">
          {!isRecording && !recordedBlob && (
            <button
              onClick={startRecording}
              className="record-button"
              disabled={cameraError}
            >
              Start Recording
            </button>
          )}

          {isRecording && (
            <button onClick={stopRecording} className="stop-button">
              Stop Recording
            </button>
          )}

          {recordedBlob && (
            <button onClick={handleAnswerComplete} className="continue-button">
              Continue
            </button>
          )}
        </div>

        <div className="recording-status">
          {isRecording && (
            <span className="recording-indicator">● Recording...</span>
          )}
          {recordedBlob && (
            <span className="recorded-indicator">✓ Recording Complete</span>
          )}
        </div>
      </div>
    );
  }

  if (page === "completed") {
    return (
      <div className="completed-page">
        <Header />

        <h2 className="page-title">Question {currentQuestion} Completed</h2>
        <p className="completion-text">
          Great job! This video is currently being analyzed. Proceed to the next
          question when you are ready!
        </p>

        {currentQuestion < totalQuestions ? (
          <button
            onClick={() => {
              setCurrentQuestion(currentQuestion + 1);
              setRecordedBlob(null); // Reset for next question
              setPage("prepare");
            }}
            className="continue-button"
          >
            Next Question
          </button>
        ) : (
          <button
            onClick={() => setPage("practiceComplete")}
            className="continue-button"
          >
            View Results
          </button>
        )}
      </div>
    );
  }

  if (page === "practiceComplete") {
    // Mock data - will be fetched from backend later
    const practiceRunNumber = 1;
    const nonVerbalScore = 50;
    const verbalScore = 50;
    const overallScore = 50;

    const CircularProgress = ({
      percentage,
      size = 120,
      strokeWidth = 8,
      color = "#22c55e",
    }) => {
      const radius = (size - strokeWidth) / 2;
      const circumference = radius * 2 * Math.PI;
      const offset = circumference - (percentage / 100) * circumference;

      return (
        <div
          className="circular-progress"
          style={{ width: size, height: size }}
        >
          <svg width={size} height={size}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke="#333"
              strokeWidth={strokeWidth}
              fill="none"
            />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={color}
              strokeWidth={strokeWidth}
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              style={{
                transition: "stroke-dashoffset 0.5s ease-in-out",
                transform: "rotate(-90deg)",
                transformOrigin: "50% 50%",
              }}
            />
          </svg>
        </div>
      );
    };

    return (
      <div className="practice-complete-page">
        <Header />

        <div className="practice-results-container">
          <h2 className="results-title">Practice Run #{practiceRunNumber}</h2>

          {/* Practice Summary */}
          <div className="practice-summary">
            <div className="summary-item">
              <strong>Topics:</strong> {practiceTopics.join(", ")}
            </div>
            <div className="summary-item">
              <strong>Number of Questions:</strong> {totalQuestions}
            </div>
            <div className="summary-item">
              <strong>Prep Time Per Question:</strong> {prepTime}
            </div>
            <div className="summary-item">
              <strong>Answer Time Per Question:</strong> {answerTime}
            </div>
          </div>

          {/* Scores Section */}
          <div className="scores-section">
            <div className="score-item">
              <h3>Non-Verbal</h3>
              <CircularProgress percentage={nonVerbalScore} />
              <div className="score-percentage">{nonVerbalScore}%</div>
            </div>
            <div className="score-item">
              <h3>Verbal</h3>
              <CircularProgress percentage={verbalScore} />
              <div className="score-percentage">{verbalScore}%</div>
            </div>
          </div>

          <div className="overall-score">
            <h3>Overall</h3>
            <CircularProgress
              percentage={overallScore}
              size={150}
              strokeWidth={10}
            />
            <div className="score-percentage">{overallScore}%</div>
          </div>

          {/* Question Feedback Tables */}
          <div className="questions-feedback">
            {Array.from({ length: totalQuestions }, (_, index) => (
              <div key={index + 1} className="question-feedback-section">
                <div className="question-video-container">
                  <h3>Question {index + 1}</h3>
                  <div className="video-placeholder">
                    {/* Placeholder for video - will be actual recorded video */}
                    <div className="mock-video">
                      <div className="play-button">▶</div>
                      <p>Recorded Video {index + 1}</p>
                    </div>
                  </div>
                </div>

                <div className="feedback-columns">
                  <div className="feedback-column">
                    <h4>Feedback on Strengths</h4>

                    <div className="feedback-category">
                      <h5>Verbal</h5>
                      <ul>
                        <li>
                          Clear articulation and confident tone throughout the
                          response
                        </li>
                        <li>
                          Good use of specific examples to support your points
                        </li>
                        <li>
                          Structured response with logical flow and conclusion
                        </li>
                      </ul>
                    </div>

                    <div className="feedback-category">
                      <h5>Non-Verbal</h5>
                      <ul>
                        <li>Maintained good eye contact with the camera</li>
                        <li>
                          Natural hand gestures that complemented your speech
                        </li>
                        <li>Confident posture and professional appearance</li>
                      </ul>
                    </div>
                  </div>

                  <div className="feedback-column">
                    <h4>Feedback on Weaknesses</h4>

                    <div className="feedback-category">
                      <h5>Verbal</h5>
                      <ul>
                        <li>
                          Could use more specific metrics or outcomes in
                          examples
                        </li>
                        <li>
                          Some filler words detected that could be reduced
                        </li>
                        <li>
                          Response could benefit from more concise phrasing
                        </li>
                      </ul>
                    </div>

                    <div className="feedback-category">
                      <h5>Non-Verbal</h5>
                      <ul>
                        <li>
                          Occasional fidgeting with hands could be minimized
                        </li>
                        <li>Could maintain more consistent eye contact</li>
                        <li>
                          Facial expressions could be more varied and engaging
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={() => setPage("practice")}
            className="return-home-button"
          >
            Return to Home
          </button>
        </div>
      </div>
    );
  }

  return null;
}

export default App;
