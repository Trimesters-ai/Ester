// Utility for streaming OpenAI chat completions
import type { ChatMessage } from '../types';

const ENV_OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;
const OPENAI_API_URL = 'https://api.openai.com/v1/responses';

// System prompt for postpartum/Whoop context
const SYSTEM_PROMPT = `You are Ester, a compassionate, medically-informed postpartum recovery assistant. You support all individuals through their postpartum journey, including those who have experienced live births, stillbirths, or pregnancy losses. Never assume the outcome of the user's pregnancy. Only reference the baby if the user has explicitly mentioned their baby's status. Never offer congratulations unless the user has explicitly shared positive news about their baby. Be especially sensitive to the full spectrum of postpartum experiences and emotions. Provide the same level of care and support regardless of pregnancy outcome. Use the user's Whoop health data (if provided) to give personalized, supportive, and clear advice. Never ask the user to provide a date in a specific format such as YYYY-MM-DD, ISO, or W3C. Always ask for and reference dates in a natural, conversational way.`;

const INSTRUCTIONS = `Format all responses in markdown for readability. Use Whoop health data context if available. Be empathetic, clear, and actionable. Be extremely mindful that users may have experienced different pregnancy outcomes, including pregnancy loss. Never assume the status of the user's baby. Only reference the baby if the user has mentioned them first. Never offer congratulations or assume a positive outcome unless explicitly stated by the user. Be particularly sensitive when discussing postpartum recovery, as the emotional and physical needs can vary greatly. Never ask the user to provide a date in a specific format such as YYYY-MM-DD, ISO, or W3C. Always ask for and reference dates in a natural, conversational way.

For new conversations, follow these steps:
1. If this is the first message in the conversation, introduce yourself warmly: "Hello, I'm Ester, your postpartum recovery assistant. I'm here to support you through your postpartum journey. May I know your name? (Only if you feel comfortable sharing)"
2. After learning their name, ask for their postpartum date in a natural way: "Thank you for sharing, [Name]. Could you tell me when your postpartum period began? This helps me provide you with the most relevant support."
3. If they don't provide a name, continue the conversation naturally without pressing for it.
4. If they don't provide a postpartum date, you can ask once more later in the conversation, but don't be insistent.
5. Remember to be sensitive and understanding if they choose not to share this information.`;

/**
 * Streams incremental response text deltas from the OpenAI chat completion API, tailored for a postpartum recovery assistant.
 *
 * Constructs a prompt using the provided conversation history, optional Whoop health data, and system instructions, then streams the assistant's response as it is generated.
 *
 * @param messages - The conversation history as an array of chat messages.
 * @param whoopData - Optional Whoop health data for personalized responses.
 * @param max_tokens - Maximum number of tokens for the response (default: 512).
 * @param temperature - Sampling temperature for response variability (default: 0.7).
 * @param user - The user object, which may contain a personal OpenAI API key.
 * @param systemPromptPrefix - Optional prefix to prepend to the system prompt.
 * @returns An asynchronous generator yielding response text deltas from the assistant.
 *
 * @throws {Error} If the OpenAI API key is missing, the API response is not OK, or the response body is absent.
 */
export async function streamOpenAIChat({
  messages,
  whoopData,
  max_tokens = 512,
  temperature = 0.7,
  user,
  systemPromptPrefix = ''
}: {
  messages: ChatMessage[];
  whoopData?: any;
  max_tokens?: number;
  temperature?: number;
  user: any;
  systemPromptPrefix?: string;
}) {
  // Prefer the user's API key if set, otherwise use the env key
  const OPENAI_API_KEY = user?.openaiApiKey || ENV_OPENAI_API_KEY;
  if (!OPENAI_API_KEY) throw new Error('Missing OpenAI API key. Please add it in your Profile page.');

  // Compose a single input string for the API
  let input = (systemPromptPrefix ? systemPromptPrefix + '\n' : '') + SYSTEM_PROMPT + '\n';
  if (whoopData) {
    input += `User's latest Whoop health data: ${JSON.stringify(whoopData)}\n`;
  }
  for (const m of messages) {
    input += (m.isAI ? 'Assistant: ' : 'User: ') + m.content + '\n';
  }

  // Get commit ID from environment variables
  const commitId = import.meta.env.VITE_COMMIT_ID || 'local-development';

  const response = await fetch(OPENAI_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      input,
      instructions: INSTRUCTIONS,
      temperature,
      stream: true,
      metadata: {
        commit: commitId
      }
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('OpenAI API error:', errorText);
    throw new Error('OpenAI API error: ' + errorText);
  }

  if (!response.body) throw new Error('No response body from OpenAI');

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let done = false;
  let buffer = '';

  async function* streamChunks() {
    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      if (value) {
        buffer += decoder.decode(value, { stream: true });
        let lines = buffer.split('\n');
        buffer = lines.pop()!;
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.replace('data: ', '').trim();
            if (data === '[DONE]') return;
            try {
              const parsed = JSON.parse(data);
              console.log('OpenAI stream chunk:', parsed);
              if (parsed.type === "response.output_text.delta" && parsed.delta) {
                yield parsed.delta;
              }
            } catch {}
          }
        }
      }
    }
  }
  return streamChunks();
}
