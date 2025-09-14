import { initializeApp } from "firebase/app";
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { Path } from "./enums";

const firebaseConfig = {
  apiKey: "AIzaSyCfJXv5r_X-t4epj3eHUkSqZMC8ZvvHImg",
  authDomain: "speakeasy-htn-25.firebaseapp.com",
  projectId: "speakeasy-htn-25",
  storageBucket: "speakeasy-htn-25.firebasestorage.app",
  messagingSenderId: "523480907377",
  appId: "1:523480907377:web:f63278a46598cb1b1b25d0",
  measurementId: "G-2GP20QXN3C",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

export const generateQuestions = (numberOfQuestions, topics) => {
  let questions = [];

  switch (topics.length) {
    case 1:
      for (let i = 0; i < numberOfQuestions; i++) {
        questions.push(topics);
      }
      break;
    case 2:
      for (let i = 0; i < numberOfQuestions; i++) {
        questions.push([topics[i % 2]]);
      }
      break;
    default:
      let start = 0;
      let len = topics.length;
      for (let i = numberOfQuestions; i > 0; i--) {
        const size = Math.ceil(len / i);
        result.push(arr.slice(start, start + size));
        start += size;
        len -= size;
      }
      return result;
      const chunkSize = Math.ceil(topics.length / numberOfQuestions); // ensures roughly equal parts

      for (let i = 0; i < len; i += chunkSize) {
        questions.push(arr.slice(i, i + chunkSize));
      }
  }

  return questions;
};

export const processVideo = async (
  videoObject,
  question,
  userId,
  topics,
  practiceRunId = -1
) => {
  const videoFile = videoObject.files;
  if (!videoFile || videoFile.length === 0) return;

  const reader = new FileReader();
  try {
    const storageRef = ref(storage, "videos/" + videoFile[0].name);
    let videoUrl = null;
    await uploadBytes(storageRef, videoFile[0]).then(async (snapshot) => {
      await getDownloadURL(snapshot.ref).then((downloadURL) => {
        videoUrl = downloadURL;
      });
    });

    if (!videoUrl) return null;

    return await analyzeVideo(
      videoUrl,
      question,
      userId,
      topics,
      practiceRunId
    );
  } catch {
    return null;
  }
};

const analyzeVideo = async (
  videoUrl,
  question,
  userId,
  topics,
  practiceRunId
) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}${Path.ANALYZE_VIDEO}?id=${userId}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ videoUrl, question, topics, practiceRunId }),
      }
    );

    if (response.ok) {
      return true;
    } else {
      return null;
    }
  } catch (error) {
    return null;
  }
};

export const addUser = async (topics) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}${Path.ADD_USER}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ topics }),
      }
    );

    if (response.ok) {
      const userIdObject = await response.json();
      return userIdObject.userId;
    } else {
      return null;
    }
  } catch (error) {
    return null;
  }
};

export const getUser = async (userId) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}${Path.GET_USER}?id=${userId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (response.ok) {
      const userObject = await response.json();
      return userObject.user;
    } else {
      return null;
    }
  } catch (error) {
    return null;
  }
};
