// Utility for streaming OpenAI chat completions via a backend proxy
import type { ChatMessage } from '../types';

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
  // Compose a single input string for the API
  let input = (systemPromptPrefix ? systemPromptPrefix + '\n' : '') + SYSTEM_PROMPT + '\n';
  if (whoopData) {
    input += `User's latest Whoop health data: ${JSON.stringify(whoopData)}\n`;
  }
  for (const m of messages) {
    input += (m.isAI ? 'Assistant: ' : 'User: ') + m.content + '\n';
  }

  const commitId = import.meta.env.VITE_COMMIT_ID || 'local-development';

  // Call backend proxy to avoid CORS / browser network issues
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages,
      whoopData,
      max_tokens,
      temperature,
      systemPromptPrefix,
      commitId,
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Backend /api/chat error:', errorText);
    throw new Error('Backend /api/chat error: ' + errorText);
  }

  const data: { text: string } = await response.json();

  async function* streamChunks() {
    // Simple non-streaming implementation: yield the full text once
    yield data.text ?? '';
  }

  return streamChunks();
}
