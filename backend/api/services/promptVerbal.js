const OpenAI = require("openai");

async function verbalPrompt(prompt) {
  const apiKey = process.env.MARTIAN_API_KEY; // Replace with your actual API key
  const requestPayload = {
    model: "openai/gpt-4.1-nano",
    messages: [
      {
        role: "system",
        content: prompt[0],
      },
      {
        role: "user",
        content: prompt[1],
      },
    ],
  };

  const response = await fetch(
    "https://api.withmartian.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(requestPayload),
    }
  );

  if (response.ok) {
    const data = await response.json();
    return data.choices[0]?.message?.content;
  } else {
    throw new Error();
  }
}

async function questionPrompt(prompt) {
  const apiKey = process.env.MARTIAN_API_KEY; // Replace with your actual API key
  const requestPayload = {
    model: "openai/gpt-4.1-nano",
    messages: [
      {
        role: "system",
        content: prompt,
      }
    ],
  };

  const response = await fetch(
    "https://api.withmartian.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(requestPayload),
    }
  );

  if (response.ok) {
    const data = await response.json();
    return data.choices[0]?.message?.content;
  } else {
    throw new Error();
  }
}

module.exports = {
  verbalPrompt,
  questionPrompt
};
