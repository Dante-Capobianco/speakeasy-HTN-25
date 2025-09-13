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

export const uploadVideoAndGetLink = async (videoObject) => {
  const videoFile = videoObject.files;
  if (!videoFile || videoFile.length === 0) return;

  const reader = new FileReader();
  reader.readAsDataURL(videoFile[0]);
  reader.onload = async () => {
    try {
      const storageRef = ref(storage, "videos/" + videoFile[0].name);
      let videoUrl = null;
      console.log(videoUrl);
      await uploadBytes(storageRef, reader.result).then(async (snapshot) => {
        console.log("hit");
        await getDownloadURL(snapshot.ref).then((downloadURL) => {
          videoUrl = downloadURL;
          console.log(downloadURL);
        });
      });

      return videoUrl;
    } catch {
      return null;
    }
  };
};

const addUser = async (topics) => {

}

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
